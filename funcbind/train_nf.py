import os, tempfile, pathlib


def _resolve_scratch_tmp() -> pathlib.Path:
    """Pick a writable scratch directory, falling back to system defaults."""
    candidates = []
    env_tmp = os.environ.get("TMPDIR")
    if env_tmp:
        candidates.append(pathlib.Path(env_tmp))
    user = os.environ.get("USER")
    if user:
        candidates.append(pathlib.Path(f"/tmp/{user}/funcbind"))
    candidates.append(pathlib.Path(tempfile.gettempdir()))

    tried = []
    for path in candidates:
        tried.append(str(path))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return path

    raise RuntimeError(f"Unable to create scratch tmp directory; attempted: {tried}")


scratch_tmp = _resolve_scratch_tmp()
os.environ.setdefault("TMPDIR", str(scratch_tmp))   # respected by C/POSIX libs
tempfile.tempdir = str(scratch_tmp)                 # respected by Python's tempfile

import logging
logging.disable(logging.CRITICAL)
import math
import os
import time
import copy
import signal
from funcbind.utils.constants import is_antibody_dataset
import hydra
import torch
import torch.nn.functional as F
import torchmetrics
from rdkit import Chem
from torch import nn
import gc
from funcbind.dataset.field_maker import FieldMaker
from funcbind.utils.utils_base import log_epoch, setup_fabric
from funcbind.utils.utils_convert import mol2rdkit_obabel
from funcbind.utils.utils_dataset import create_field_loaders
from funcbind.utils.utils_nf import (
    create_neural_field,
    infer_codes_batch,
    infer_occupancies_batch,
    get_points,
    load_neural_field,
    occs_to_mol,
    plot_xs,
    save_checkpoint,
)
from funcbind.utils.utils_vis import convert_sdf_to_pdb, visualize_ligand_receptor
from funcbind.utils.utils_sampling import render_samples


def compute_loss(pred, occs, criterion):
    """
    Compute loss based on the specified loss type.

    Args:
        pred: Predicted values
        occs: Target occupancy values
        criterion: Loss criterion (nn.MSELoss, nn.L1Loss, etc.)

    Returns:
        Computed loss
    """
    return criterion(pred, occs)


@hydra.main(config_path="configs", config_name="train_nf", version_base=None)
def main(config):
    fabric = setup_fabric(config)
    field_maker = FieldMaker(config)
    field_maker = field_maker.to(fabric.device)

    ##############################
    # model
    acc_iter = 0
    if config["nf_pretrained_path"] is not None and os.path.exists(os.path.join(config["nf_pretrained_path"], "model.pt")):
        checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
        enc, dec = load_neural_field(checkpoint, fabric, config if "overwrite_config" in config and config["overwrite_config"] else checkpoint["config"], setup_fabric=False)
        checkpoint_optim_enc, checkpoint_optim_dec = checkpoint["optim_enc"], checkpoint["optim_dec"]
        acc_iter = checkpoint.get("acc_iter", 0)
    else:
        enc, dec = create_neural_field(config, fabric)
        checkpoint_optim_enc, checkpoint_optim_dec = None, None
    enc.device = fabric.device
    dec.device = fabric.device

    criterion = nn.MSELoss()

    # optimizers
    optim_enc = torch.optim.Adam([{"params": enc.parameters(), "lr": config["dset"]["lr_enc"]}])
    optim_dec = torch.optim.Adam([{"params": dec.parameters(), "lr": config["dset"]["lr_dec"]}])

    # fabric setup
    enc, optim_enc = fabric.setup(enc, optim_enc)
    dec, optim_dec = fabric.setup(dec, optim_dec)

    # Load optimizer
    if checkpoint_optim_enc is not None and checkpoint_optim_dec is not None:
        fabric.print(">> loading optimizer state")
        optim_enc.load_state_dict(checkpoint_optim_enc)
        optim_dec.load_state_dict(checkpoint_optim_dec)

    ##############################
    # data loaders
    loader_train = create_field_loaders(
        config, split="train", fabric=fabric, n_samples=config.get("n_samples")
    )
    loader_val = create_field_loaders(
        config, split="val", fabric=fabric, n_samples=config.get("n_samples")
    )

    config_rec = copy.deepcopy(config)
    config_rec["dset"]["batch_size"] = 1  # for reconstruction we use batch size of 1
    loader_eval_rec = create_field_loaders(config_rec, split="val", fabric=fabric, sample_full_grid=True, shuffle=True)

    ##############################
    # metrics
    metrics = {
        "loss": torchmetrics.MeanMetric().to(fabric.device),
        "loss_rec": torchmetrics.MeanMetric().to(fabric.device),
        "loss_reg": torchmetrics.MeanMetric().to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex().to(fabric.device),
    }
    metrics_val = {
        "loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
        "loss_rec": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
        "loss_reg": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex(sync_on_compute=False).to(fabric.device),
    }

    ##############################
    # start training the neural field
    fabric.print(">> start training the neural field", config["exp_name"])
    best_loss = None  # save checkpoint each time save_checkpoint is called

    for epoch in range(config["n_epochs"]):
        start_time = time.time()

        # train
        loss_train, miou_train, loss_rec_train, loss_reg_train, acc_iter = train_nf(
            loader_train, dec, optim_dec, enc, optim_enc, criterion, fabric, metrics=metrics, field_maker=field_maker, config=config, acc_iter=acc_iter,
        )

        # val
        loss_val, miou_val, loss_rec_val, loss_reg_val = None, None, None, None
        try:
            loss_val, miou_val, loss_rec_val, loss_reg_val = val_nf(
                loader_val, dec, enc, criterion, config, metrics=metrics_val, field_maker=field_maker, fabric=fabric,
            )
        except Exception as e:
            fabric.print(f"Error during validation: {e}")
        try:
            best_loss = save_checkpoint(
                epoch, config, loss_val, best_loss, enc, dec, optim_enc, optim_dec, fabric, acc_iter,
            )
        except Exception as e:
            fabric.print(f"Error saving checkpoint: {e}")

        # eval reconstuction
        if epoch != 0 and (epoch % config["eval_every"] == 0 or epoch == config["n_epochs"] - 1):
            # Set up 20-minute timeout for eval_nf_rec
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(20 * 60)  # 20 minutes in seconds

            try:
                eval_nf_rec(
                    loader_eval_rec, dec, enc, config, epoch, field_maker=field_maker, fabric=fabric, n_samples=[0,1,2,3],
                )
            except TimeoutException as e:
                fabric.print(f"Warning: {e}")
                fabric.print("Skipping evaluation and continuing with training...")
            finally:
                # Reset the alarm and restore the previous signal handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        # log
        elapsed_time = time.time() - start_time
        log_epoch(
            config, epoch, loss_train, miou_train, loss_val, miou_val, elapsed_time, fabric, loss_rec_train=loss_rec_train,
            loss_reg_train=loss_reg_train, loss_rec_val=loss_rec_val, loss_reg_val=loss_reg_val, acc_iter=acc_iter,
            optimizer_enc=optim_enc, optimizer_dec=optim_dec
        )

        if math.isnan(loss_train):
            fabric.print(">> train_loss is nan. Exiting...")
            break


def train_nf(
    loader, dec, optim_dec, enc, optim_enc, criterion, fabric, metrics, field_maker, config, acc_iter=0,
):
    enc.train()
    dec.train()
    for key in metrics.keys():
        metrics[key].reset()

    for i, batch in enumerate(loader):
        # get the points xs according to the encoder type
        xs = get_points(batch)

        # compute codes (forward INR encoder), their posteriors, points xs and their corresponding ground truth occupancies
        codes, posteriors = infer_codes_batch(batch, enc, field_maker, config, xs=xs)
        occs = infer_occupancies_batch(batch, field_maker, config)

        # forward NF decoder
        pred = dec(xs, codes)
        loss = compute_loss(pred, occs, criterion)
        metrics["loss_rec"].update(loss)
        metrics["miou"].update((pred > 0.5).to(torch.uint8), (occs > 0.5).to(torch.uint8))

        acc_iter += codes.size(0)

        # regularization
        if config["reg_weight"] != 0.0:
            reg_loss = posteriors.kl()
            reg_loss = config["reg_weight"] * torch.sum(reg_loss) / reg_loss.shape[0]
            metrics["loss_reg"].update(reg_loss)
            loss += reg_loss
        metrics["loss"].update(loss)

        # backward
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        fabric.backward(loss)

        optim_enc.step()
        optim_dec.step()

        if math.isnan(loss):
            fabric.print(">> train_loss is nan. Exiting...")
            break

    return metrics["loss"].compute().item(), metrics["miou"].compute().item(), \
        metrics["loss_rec"].compute().item(), metrics["loss_reg"].compute().item(), acc_iter


@torch.no_grad()
def val_nf(
    loader, dec, enc, criterion, config, metrics, field_maker, fabric, plot_xs_flag=False, sample_full_grid=False,
):
    dec = dec.module
    enc = enc.module
    dec.eval()
    enc.eval()
    for key in metrics.keys():
        metrics[key].reset()

    for i, batch in enumerate(loader):
        # get the points xs according to the encoder type
        xs = get_points(batch)
        if plot_xs_flag and i == 0:
            fabric.print(f">> plotting xs shape: {xs.shape}")
            for j in range(min(xs.shape[0], 10)):
                plot_xs(xs[j].cpu(), j, output_dir=os.path.join(config["dirname"], "output_plots"))

        # compute codes (forward INR encoder), their posteriors, points xs and their corresponding ground truth occupancies
        codes, posteriors = infer_codes_batch(batch, enc, field_maker, config, xs=xs, save_log_var=not sample_full_grid)
        occs = infer_occupancies_batch(batch, field_maker, config)

        # Forward pass through INR decoder
        if sample_full_grid:
            occs = occs.cpu()
            batch_size_render = 150_000
            pred = dec.forward_batched(xs, codes, batch_size_render=batch_size_render, to_cpu=True)
            del xs, codes, posteriors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            pred = dec(xs, codes)
        loss = compute_loss(pred, occs, criterion)
        metrics["loss_rec"].update(loss)
        metrics["miou"].update((pred > 0.5).to(torch.uint8), (occs > 0.5).to(torch.uint8))

        # Regularization
        reg_loss = None
        if config["reg_weight"] != 0.0 and not sample_full_grid:
            reg_loss = posteriors.kl()
            reg_loss = config["reg_weight"] * torch.sum(reg_loss) / reg_loss.shape[0]
            metrics["loss_reg"].update(reg_loss)
            loss += reg_loss
        metrics["loss"].update(loss)

    return metrics["loss"].compute().item(), metrics["miou"].compute().item(), \
        metrics["loss_rec"].compute().item(), metrics["loss_reg"].compute().item()


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("eval_nf_rec timed out after 20 minutes")


@torch.no_grad()
def eval_nf_rec(loader, dec, enc, config, epoch, field_maker, fabric, n_samples=None, deterministic=False, refine=False, plot=False):
    dirname = os.path.join(config["dirname"], "reconstruction", f"epoch_{epoch:02d}")
    os.makedirs(dirname, exist_ok=True)

    dec = dec.module
    enc = enc.module
    dec.eval()
    enc.eval()

    for i, batch in enumerate(loader):
        if (n_samples is not None and i in n_samples) or n_samples is None:
            try:
                # infer codes and mols
                preds, mols = None, None
                if refine:
                    codes, _ = infer_codes_batch(batch, enc, field_maker, config, save_log_var=False, deterministic=deterministic)
                    mols = render_samples(codes, dec, fabric, config, unnormalize=False)
                else:
                    xs = dec.coords.to(fabric.device)
                    codes, _ = infer_codes_batch(batch, enc, field_maker, config, xs=xs, save_log_var=False, deterministic=deterministic)
                    batch_size_render = 150_000
                    preds = dec.forward_batched(xs, codes, batch_size_render=batch_size_render, to_cpu=True)
                    grid_dim_infer = config["dset"]["grid_dim"]
                    preds = preds.permute(0, 2, 1).reshape(-1, preds.size(2), grid_dim_infer, grid_dim_infer, grid_dim_infer)
                    del xs
                convert_grids_to_sdf(preds, dirname, f"pred_{i}", config, fabric, mols=mols)

                del codes
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # infer gt occupancies
                occs = infer_occupancies_batch(batch, field_maker, config)
                grid_dim = config["dset"]["grid_dim"]
                occs = occs.permute(0, 2, 1).reshape(-1, occs.size(2), grid_dim, grid_dim, grid_dim)
                convert_grids_to_sdf(occs, dirname, f"gt_{i}", config, fabric)

                # plot voxels
                if plot:
                    save_grids(occs, preds, dirname, i, config)

            except Exception as e:
                fabric.print(f"Error during reconstruction for complex {i}: {e}")
                continue


def save_grids(occs, preds, dirname, idx, config):
    dirname = os.path.join(dirname, "grids")
    os.makedirs(dirname, exist_ok=True)
    if occs is not None:
        visualize_ligand_receptor(ligand=occs[0], receptor=None, name=f"lig_gt_{idx}", sparse=True, dirname=dirname)
    if preds is not None:
        visualize_ligand_receptor(ligand=preds[0], receptor=None, name=f"lig_pred_{idx}", sparse=True, dirname=dirname)


def convert_grids_to_sdf(occs, dirname, prefix, config, fabric, mols=None):
    dirname = os.path.join(dirname, "sdf")
    os.makedirs(dirname, exist_ok=True)
    if mols is None:
        mols = []
        occs_to_mol(config, fabric, vox_point=occs, mols=mols, resolution=config["dset"]["resolution"])

    len_mols = 1
    if len(mols) < len_mols:
        fabric.print("Error converting mol to rdkit")
        return
    if (mols[0]['coords'].shape[1] > 1500) or (len(mols) == 2 and mols[1]['coords'].shape[1] > 3000):
        fabric.print("Molecule is too large to convert to SDF. Skipping...", mols[0]['coords'].shape[1], mols[1]['coords'].shape[1] if len(mols) == 2 else None)
        return

    for i, mol in enumerate(mols):
        fname = f"lig_{prefix}" if i % 2 == 0 else f"rec_{prefix}"
        try:
            rdmol = mol2rdkit_obabel(mol, remove_fragment=False, remove_atoms_too_close=config["sampling"]["remove_atoms_too_close"])
            w = Chem.SDWriter(os.path.join(dirname, f"{fname}.sdf"))
            w.write(rdmol)
            w.close()
            if is_antibody_dataset(config):
                convert_sdf_to_pdb(dirname, fabric=fabric, fname=f"{fname}.pdb", fname_sdf=f"{fname}.sdf")
        except Exception as e:
            fabric.print(f"{e} error when converting mol index {i} to rdkit")
            rdmol = None


if __name__ == "__main__":
    main()
