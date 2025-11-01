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

# Set PyTorch CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import math
import time

from funcbind.models.phema import PowerFunctionEMA
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm
from hydra.core.global_hydra import GlobalHydra
from funcbind.utils.constants import is_antibody_dataset

from funcbind.distributions.distributions import Normal, UniformPlusNormal
from funcbind.models.denoiser import add_noise_to_code, log_metrics
from funcbind.utils.constants import N_RECEPTOR_ELEMENTS
from funcbind.utils.utils_base import makedir, setup_fabric
from funcbind.utils.utils_dataset import create_field_loaders
from funcbind.utils.utils_fb import (
    create_field_makers,
    create_funcbind,
    create_optimizer,
    learning_rate_schedule,
    load_funcbind,
    num_classes_funcbind,
)
from funcbind.utils.utils_metrics import create_sampling_metrics, evaluate_target_metrics, log_results
from funcbind.utils.utils_nf import (
    filter_mol_to_sdf_pdb,
    infer_codes_batch,
    load_neural_field,
    update_config_nf,
)
from funcbind.utils.utils_sampling import (
    filter_valid_mol,
    get_ligand_encoding,
    get_receptor_encoding,
    random_rot_matrix,
    render_samples,
    save_samples,
)


@hydra.main(config_path="configs", config_name="train_fb", version_base=None)
def main(config):
    fabric = setup_fabric(config)

    exp_name, dirname = config["exp_name"], config["dirname"]
    config = OmegaConf.to_container(config)
    config["exp_name"], config["dirname"] = exp_name, dirname  # TODO: make this better
    makedir(dirname)
    val_save_dir = os.path.join(config["dirname"], "validation_plots")
    makedir(val_save_dir)
    fabric.print(">> saving experiments in:", config["dirname"])

    ##############################
    # load pretrained neural field
    ##############################
    nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
    enc, dec = load_neural_field(nf_checkpoint, fabric)
    dec_module = dec.module if hasattr(dec, "module") else dec

    ##############################
    # config update
    ##############################

    # nf and funcbind config update
    config_nf = update_config_nf(nf_checkpoint["config"], config)
    config["decoder"] = config_nf["decoder"]
    config["encoder"] = config_nf["encoder"]

    # val config
    config_val = copy.deepcopy(config)
    plot_bs = 10
    config_val["sampling"]["batch_size_render"] *= 2 * (config_val["sampling"]["n_chains"] // plot_bs)  # bump up rendering batch size

    # sampling config
    # config_sampling = copy.deepcopy(config_nf)
    # config_sampling["dset"]["batch_size"] = 1

    num_classes = num_classes_funcbind(config)

    ##############################
    # field loaders
    ##############################
    loader_train = create_field_loaders(config_nf, split="train", fabric=fabric, sample_points=False, n_samples=config_nf["n_samples"])

    config_nf_val = copy.deepcopy(config_nf)
    config_nf_val["dset"]["batch_size"] = 64  # for easy comparison across batch sizes
    config_val["dset"]["batch_size"] = 64
    loader_val = create_field_loaders(config_nf_val, split="val", fabric=fabric, sample_points=False, drop_last=False) if fabric.global_rank == 0 else None
    # loader_sampling = create_field_loaders(config_sampling, split=config["sampling"]["split"], fabric=fabric, sample_points=False, shuffle=False) if fabric.global_rank == 0 else None

    ##############################
    # create field maker
    ##############################
    field_maker, field_maker_receptor = create_field_makers(config, config_nf, fabric)

    ##############################
    # code stats
    ##############################
    acc_iter = 0
    checkpoint_optimizer = None
    if config["fb_pretrained_path"] is not None and os.path.exists(os.path.join(config["fb_pretrained_path"], "checkpoint.pth.tar")):
        fabric.print(f">> loading checkpoint from {config['fb_pretrained_path']}")
        funcbind, funcbind_ema, checkpoint_optimizer, code_stats, acc_iter = load_funcbind(
            config["fb_pretrained_path"], fabric=fabric, config=config, num_classes=num_classes, train=True
        )
    else:
        code_stats = compute_code_stats(
            loader_train, enc, config_nf, "train", fabric, True, field_maker=field_maker, save_dir=val_save_dir, debug=config["debug"],
        )
        funcbind = create_funcbind(config, code_stats=code_stats, fabric=fabric, num_classes=num_classes)

        # EMA
        with torch.no_grad():
            assert "ema_stds" in config and config["ema_stds"] is not None and (len(config["ema_stds"]) > 0 and config["ema_stds"][0] != 0.0), "ema_stds must be set and non-empty"
            fabric.print(">> using PowerFunctionEMA with stds", config["ema_stds"])
            funcbind_ema = PowerFunctionEMA(funcbind, stds=config["ema_stds"])
    dec_module.code_stats = code_stats

    ##############################
    # optimizer and fabric
    ##############################
    optimizer = create_optimizer(funcbind, config, fabric)
    funcbind, optimizer = fabric.setup(funcbind, optimizer)
    if checkpoint_optimizer is not None:
        fabric.print(">> loading optimizer state")
        optimizer.load_state_dict(checkpoint_optimizer)

    ##############################
    # metrics
    ##############################
    metrics = torchmetrics.MeanMetric().to(fabric.device)
    if config["sampler"]["val_sigmas"] is not None:
        val_sigmas = np.array(config["sampler"]["val_sigmas"])
    else:
        val_sigmas = funcbind.sigma_distribution.values.numpy()
    metrics_val = {}
    for sigma in val_sigmas:
        metrics_val[sigma] = {
            "weighted_loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
            "loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
            "weight_over_var": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
            "logvar": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
        }

    ##############################
    # start training
    ##############################
    fabric.print(">> start training the denoiser", config["exp_name"])
    best_res = 1e10

    for epoch in range(0, config["num_epochs"]):
        t0 = time.time()

        # train
        train_loss, acc_iter = train_denoiser(
            loader_train,
            enc,
            dec_module,
            funcbind,
            optimizer,
            metrics,
            config,
            config_nf,
            funcbind_ema,
            acc_iter,
            fabric,
            field_maker=field_maker,
            field_maker_receptor=field_maker_receptor,
            num_classes=num_classes,
        )

        # val
        val_loss = None
        val_weighted_loss = None
        val_losses, val_weighted_losses, val_weights_over_vars, val_logvars = (
            [None for _ in range(len(val_sigmas))],
            [None for _ in range(len(val_sigmas))],
            [None for _ in range(len(val_sigmas))],
            [None for _ in range(len(val_sigmas))],
        )
        val_validity = {}
        if (epoch + 1) % config["val_every"] == 0:
            with fabric.rank_zero_first():
                if fabric.global_rank == 0:
                    (
                        val_weighted_losses,
                        val_losses,
                        val_weights_over_vars,
                        val_logvars,
                        val_validity,
                    ) = val_denoiser(
                        loader_val,
                        enc,
                        dec_module,
                        funcbind_ema,
                        metrics_val,
                        config_val,
                        config_nf,
                        field_maker=field_maker,
                        field_maker_receptor=field_maker_receptor,
                        fabric=fabric,
                        plot_bs=plot_bs,
                        num_classes=num_classes,
                    )
                    val_loss = sum(val_losses) / len(val_losses)
                    val_weighted_loss = sum(val_weighted_losses) / len(val_weighted_losses)
                    if val_weighted_loss <= best_res:
                        best_res = val_weighted_loss
                        state = {
                            "epoch": epoch + 1,
                            "config": config,
                            "state_dict": funcbind.state_dict(),
                            "state_dict_ema": funcbind_ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "code_stats": dec_module.code_stats,
                            "acc_iter": acc_iter,
                        }
                        checkpoint_path = os.path.join(config["dirname"], "checkpoint.pth.tar")
                        try:
                            torch.save(state, checkpoint_path)
                            fabric.print(f"New best checkpoint saved: {checkpoint_path}, best_res: {best_res}")
                        except Exception as e:
                            fabric.print(f"Error saving checkpoint: {e}")

                    plot_metric_vs_sigma(
                        val_sigmas,
                        val_losses,
                        "Validation Loss",
                        y_label="Loss",
                        color="red",
                        marker="s",
                        save_dir=val_save_dir,
                    )
                    plot_metric_vs_sigma(
                        val_sigmas,
                        val_weighted_losses,
                        "Weighted Validation Loss",
                        y_label="Weighted Loss",
                        color="green",
                        marker="^",
                        save_dir=val_save_dir,
                    )
                    plot_metric_vs_sigma(
                        val_sigmas,
                        val_weights_over_vars,
                        "Weight over Variance",
                        y_label="Weight / Variance",
                        color="purple",
                        marker="d",
                        save_dir=val_save_dir,
                    )
                    plot_metric_vs_sigma(
                        val_sigmas,
                        val_logvars,
                        "Log Variance",
                        y_label="Log Variance",
                        color="orange",
                        marker="*",
                        save_dir=val_save_dir,
                    )
                    plot_metric_vs_sigma(
                        val_sigmas,
                        [val_validity[sigma_] for sigma_ in val_sigmas],
                        "Validity",
                        y_label="Validity Score",
                        color="cyan",
                        marker="x",
                        save_dir=val_save_dir,
                    )

        # sample molecules
        sampling_metrics = None
        # if ((epoch + 1) % config["sample_every"] == 0 or epoch == config["num_epochs"] - 1) and epoch != 0:
        #     with fabric.rank_zero_first():
        #         if fabric.global_rank == 0:
        #             if epoch == config["num_epochs"] - 1:
        #                 fabric.print(">> sampling 1k on the last epoch.")
        #                 config["sampling"]["n_attempts"] = 20
        #             try:
        #                 sampling_metrics = sample(loader_sampling, funcbind_ema, enc, dec_module, config=config, config_nf=config_nf, fabric=fabric, field_maker=field_maker, field_maker_receptor=field_maker_receptor, num_classes=num_classes)
        #             except Exception as e:
        #                 fabric.print(f"Error during sampling: {e}")
        #                 sampling_metrics = None

        # log metrics
        log_metrics(
            config["exp_name"],
            epoch,
            train_loss,
            val_loss,
            sampling_metrics,
            best_res,
            time.time() - t0,
            fabric,
        )

        if config["wandb"]:
            fabric.log_dict(
                {
                    "trainer/global_step": epoch,
                    "acc_iter": acc_iter,
                    "acc_iter_normalized": acc_iter / (config["dset"]["batch_size"] * fabric.world_size),
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_weighted_loss": val_weighted_loss,
                    **{f"val_loss_sigma_{sigma}": loss for sigma, loss in zip(val_sigmas, val_losses)},
                    **{f"val_weighted_loss_sigma_{sigma}": loss for sigma, loss in zip(val_sigmas, val_weighted_losses)},
                    **{f"val_weight_over_var_sigma_{sigma}": loss for sigma, loss in zip(val_sigmas, val_weights_over_vars)},
                    **{f"val_logvar_sigma_{sigma}": loss for sigma, loss in zip(val_sigmas, val_logvars)},
                    **{f"val_validity_sigma_{sigma}": validity for sigma, validity in val_validity.items()},
                    "sampling": sampling_metrics,
                }
            )

        # If train_loss is nan, break
        if math.isnan(train_loss):
            fabric.print(">> train_loss is nan. Exiting...")
            break


def train_denoiser(
    loader,
    enc,
    dec_module,
    model,
    optimizer,
    metrics,
    config,
    config_nf,
    model_ema=None,
    acc_iter=0,
    fabric=None,
    field_maker=None,
    field_maker_receptor=None,
    num_classes=None
):

    """
    Train a denoising model using the provided data loader, model, and training configuration.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the training data.
        enc (torch.nn.Module): Encoder module.
        dec_module (torch.nn.Module): Decoder module.
        model (torch.nn.Module): The denoising model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        metrics (torchmetrics.MeanMetric): Metric to track the training performance.
        config (dict): Configuration dictionary containing training parameters.
        model_ema (ModelEma, optional): Exponential moving average model. Defaults to None.
        acc_iter (int, optional): Accumulated iteration count. Defaults to 0.
        fabric (object, optional): Fabric object for distributed training. Defaults to None.
        field_maker (FieldMaker, optional): FieldMaker object for generating fields. Defaults to None.

    Returns:
        tuple: A tuple containing the computed metric value and the updated accumulated iteration count.
    """
    metrics.reset()
    model.train()

    for batch in loader:
        learning_rate_schedule(optimizer, acc_iter, config, world_size=fabric.world_size)

        with torch.no_grad():
            codes_ligand = infer_codes_batch(
                batch, enc, field_maker, config_nf, code_stats=dec_module.code_stats, save_log_var=False
            )[0]
            voxels_receptor = field_maker_receptor.compute_voxel_grid(
                batch["receptor"], num_channels=N_RECEPTOR_ELEMENTS
            )

            batch_size = codes_ligand.size(0)
            sigma = model.sigma_distribution.sample((batch_size,)).to(fabric.device)
            smooth_codes_ligand = add_noise_to_code(codes_ligand, sigma=sigma)

            label_batch = get_label(batch, num_classes=num_classes, config=config)

        acc_iter += (batch_size * fabric.world_size)
        if config["sampler"]["name"] == "walkjump":
            codes_pred = model(
                smooth_codes_ligand, receptor=voxels_receptor, sigma=sigma, classes=label_batch
            )
            weighted_loss = ((codes_pred - codes_ligand) ** 2).sum()
        else:
            codes_pred, logvar = model(
                smooth_codes_ligand, receptor=voxels_receptor, sigma=sigma, classes=label_batch, return_logvar=True,
            )
            sigma = sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight = (sigma**2 + model.sigma_data**2) / (
                (sigma * model.sigma_data) ** 2
            )  # Eq. 15 in EDM2
            weight_over_var = weight / logvar.exp()  # Eq. 21 in EDM2
            weighted_loss = (
                weight_over_var * ((codes_pred - codes_ligand) ** 2) + logvar
            ).sum()
        optimizer.zero_grad()
        fabric.backward(weighted_loss)
        optimizer.step()

        if hasattr(model_ema, "emas"):
            model_ema.update(acc_iter, batch_size * fabric.world_size)
        else:
            model_ema.update(model)
        metrics.update(weighted_loss)

    return metrics.compute().item(), acc_iter


@torch.no_grad()
def val_denoiser(
    loader,
    enc,
    dec_module,
    model,
    metrics,
    config,
    config_nf,
    field_maker=None,
    field_maker_receptor=None,
    fabric=None,
    plot_bs=10,
    num_classes=None,
):
    """
    Validate the denoising model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        enc (torch.nn.Module): Encoder module.
        dec_module (torch.nn.Module): Decoder module.
        model (torch.nn.Module): Denoising model.
        metrics (torchmetrics.MeanMetric): Metric to compute the mean loss.
        config (dict): Configuration dictionary containing various settings.
        field_maker (FieldMaker, optional): Optional FieldMaker instance for on-the-fly code generation.

    Returns:
        float: Computed mean loss over the validation dataset.
    """
    if hasattr(model, "emas"):
        model = model.emas[0]
    else:
        model = model.module
    model.eval()
    validity = {}

    for sigma_ in tqdm(metrics.keys()):
        for key in metrics[sigma_].keys():
            metrics[sigma_][key].reset()
        for i, batch in enumerate(loader):
            codes_ligand = infer_codes_batch(
                batch, enc, field_maker, config_nf, code_stats=dec_module.code_stats, to_cpu=False, save_log_var=False
            )[0]
            voxels_receptor = field_maker_receptor.compute_voxel_grid(
                batch["receptor"], num_channels=N_RECEPTOR_ELEMENTS
            )
            sigma = sigma_ * torch.ones(
                codes_ligand.shape[0],
                device=codes_ligand.device,
                dtype=codes_ligand.dtype,
            )
            label_batch = get_label(batch, num_classes=num_classes, config=config)
            smooth_codes_ligand = add_noise_to_code(codes_ligand, sigma=sigma)
            if config["sampler"]["name"] == "walkjump":
                codes_pred = model(
                    smooth_codes_ligand, receptor=voxels_receptor, sigma=sigma, classes=label_batch, cfg_dropout=False
                )
                weighted_loss = ((codes_pred - codes_ligand) ** 2).sum()
                loss = weighted_loss
                weight_over_var = torch.ones_like(weighted_loss)
                logvar = torch.zeros_like(weighted_loss)
            else:
                codes_pred, logvar = model(
                    smooth_codes_ligand, receptor=voxels_receptor, sigma=sigma, classes=label_batch, return_logvar=True, cfg_dropout=False
                )
                sigma = (sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                weight = (sigma**2 + model.sigma_data**2) / ((sigma * model.sigma_data) ** 2)  # Eq. 15 in EDM2
                weight_over_var = weight / logvar.exp()  # Eq. 21 in EDM2
                squared_error = (codes_pred - codes_ligand) ** 2
                loss = squared_error.sum()
                weighted_loss = (weight_over_var * squared_error + logvar).sum()

            metrics[sigma_]["weighted_loss"].update(weighted_loss)
            metrics[sigma_]["loss"].update(loss)
            metrics[sigma_]["weight_over_var"].update(weight_over_var.mean())
            metrics[sigma_]["logvar"].update(logvar.mean())

            # Clean up intermediate tensors
            del voxels_receptor, smooth_codes_ligand
            try:
                del squared_error, weight
            except NameError:
                pass

            if i == 0:
                try:
                    dirname_rec = os.path.join(config["dirname"], f"rec_{sigma_}")
                    dirname_gt = os.path.join(config["dirname"], "gt")
                    makedir(dirname_rec)
                    makedir(dirname_gt)
                    size_rec = min(plot_bs, codes_pred.size(0))
                    center_coords = batch["receptor"]["center_coords"][:size_rec].cpu()
                    codes_pred = dec_module.unnormalize_code(codes_pred[:size_rec])
                    mols_rec = dec_module.codes_to_molecules(
                        codes_pred, False, config, fabric, verbose=False, voxel_name="pred"
                    )
                    seq_pred = filter_mol_to_sdf_pdb(
                        config,
                        fabric,
                        mols_rec,
                        fname="molecules_rec",
                        dirname=dirname_rec,
                        center_coords=center_coords,
                        verbose=False,
                    )
                    validity[sigma_] = len(seq_pred) / size_rec

                    if not os.path.exists(f"{dirname_gt}/molecules_gt.sdf"):
                        codes_ligand = dec_module.unnormalize_code(codes_ligand[:size_rec])
                        mols_gt = dec_module.codes_to_molecules(
                            codes_ligand, False, config, fabric, verbose=False, voxel_name="gt"
                        )
                        seq_gt = filter_mol_to_sdf_pdb(
                            config,
                            fabric,
                            mols_gt,
                            fname="molecules_gt",
                            dirname=dirname_gt,
                            center_coords=center_coords,
                            verbose=False,
                        )
                        validity[0] = len(seq_gt) / size_rec
                except Exception as e:
                    validity[sigma_] = 0.0
                    validity[0] = 0.0
                    fabric.print(f"Error saving molecules: {e}")
                finally:
                    # Clean up remaining tensors
                    try:
                        del codes_ligand, codes_pred, logvar, weight_over_var
                    except NameError:
                        pass
                    torch.cuda.empty_cache()
            else:
                # Clean up for non-plotting batches
                try:
                    del codes_ligand, codes_pred, logvar, weight_over_var
                except NameError:
                    pass

        # Clean up memory between sigma values to prevent accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        [metrics[sigma_]["weighted_loss"].compute().item() for sigma_ in metrics.keys()],
        [metrics[sigma_]["loss"].compute().item() for sigma_ in metrics.keys()],
        [metrics[sigma_]["weight_over_var"].compute().item() for sigma_ in metrics.keys()],
        [metrics[sigma_]["logvar"].compute().item() for sigma_ in metrics.keys()],
        validity,
    )


def prepare_sampler(
    config,
    config_nf,
    fabric,
    receptor,
    ligand_gt,
    enc,
    model,
    field_maker,
    field_maker_receptor,
):
    # rotations
    if config["sampling"]["rotate_receptor"]:
        rand_rots = [random_rot_matrix() for _ in range(config["sampling"]["n_chains"])]
        fabric.print(f">> rotate gt ligand and receptor. {len(rand_rots)} rotation matrices")
    else:
        rand_rots = None
    receptor_encoding = get_receptor_encoding(receptor, model, field_maker_receptor, fabric, rand_rots, config["sampling"]["n_chains"])
    ligand_encoding = None
    if config["sampling"]["chain_init"] == "ligand":
        ligand_encoding = get_ligand_encoding(ligand_gt, model, enc, field_maker, fabric, config_nf, rand_rots, config["sampling"]["n_chains"])
        ligand_encoding = add_noise_to_code(ligand_encoding, sigma=config["sampler"]["sigma"] if config["sampler"]["name"] != "diffusion" else config["sampler"]["sigma_max"])

    return ligand_encoding, receptor_encoding, rand_rots


@torch.no_grad()
def sample(
    loader,
    model,
    enc,
    dec_module,
    config,
    config_nf,
    fabric,
    field_maker=None,
    field_maker_receptor=None,
    use_metrics=True,
    num_classes=None,
    uniform_plus_normal=True
):
    """
    Samples data using the provided model and decoder module, computes metrics, and saves the samples.
    """
    if hasattr(model, "emas"):
        model = model.emas[0]
    else:
        model = model.module
    enc = enc.module if hasattr(enc, "module") else enc
    model.eval()

    dirname_out = os.path.join(config["dirname"], "samples")
    os.makedirs(dirname_out, exist_ok=True)
    n_targets = 0

    sampler = hydra.utils.instantiate(config)["sampler"]
    if uniform_plus_normal:
        sampler.y_init_distribution = UniformPlusNormal(
            model.code_stats,
            config,
            device=fabric.device,
            dtype=torch.float32,
        )
    else:
        sampler.y_init_distribution = Normal(
            config,
            device=fabric.device,
            dtype=torch.float32,
        )
    if use_metrics:
        metrics_full = create_sampling_metrics(config, dirname_out)
        metrics_full.reset()
    metrics_sampling = None

    is_ab = is_antibody_dataset(config)
    is_mcpp = (config["dset"]["input_dataset"] == "omni_v1" and "mcpp" in config["dset"]["use_single_dataset"])
    t0 = time.time()

    for receptor_id, batch in enumerate(loader):
        receptor, ligand_gt = batch["receptor"], batch["ligand"]
        if config["sampling"]["receptor_ids"] is not None and (receptor_id not in config["sampling"]["receptor_ids"] and receptor["id"][0] not in config["sampling"]["receptor_ids"]):
            continue

        fabric.print("================================================")
        seq_seed_h3 = ligand_gt["cdr_h3_seq"][0] if is_ab else None
        df_mol = None

        fabric.print(f"| sampling receptor {receptor['id'][0]} #{receptor_id}")

        target_dirname = os.path.join(dirname_out, f"target_{receptor_id:02d}")
        os.makedirs(target_dirname, exist_ok=True)

        label_batch = get_label(batch, num_classes=num_classes, config=config)

        attempts = 0
        n_attempts = (config["sampling"]["n_attempts"] if config["sampling"]["remove_fragment"] else 1)
        valid_mols = []
        valid_seq = []

        while (len(valid_mols) < config["sampling"]["n_samples_per_receptor"] and attempts < n_attempts):
            # Step 0 : Prepare sampler
            ligand_encoding, receptor_encoding, rand_rots = prepare_sampler(
                config, config_nf, fabric, receptor, ligand_gt, enc, model, field_maker, field_maker_receptor
            )

            # Step 1: Sampling n_chains
            mols = sample_population(
                model, sampler, dec_module, config, fabric, ligand_encoding, receptor_encoding, rand_rots, label_batch=label_batch, attempts=attempts, n_attempts=n_attempts,
                target_dirname=target_dirname
            )

            # Step 2: Filter valid molecules
            valid_mols, valid_seq, df_mol = filter_valid_mol(
                mols, center_coords=receptor["center_coords"].cpu(), remove_fragment=config["sampling"]["remove_fragment"], valid_mols=valid_mols, valid_seq=valid_seq, attempts=attempts, fabric=fabric, is_ab=is_ab, unique_only=config["sampling"]["unique_only"], len_seq=len(seq_seed_h3) if (config["sampling"]["filter_length"] and seq_seed_h3 is not None) else None, remove_atoms_too_close=config["sampling"]["remove_atoms_too_close"], is_mcpp=is_mcpp, config=config, target_dirname=config["save_dir"], df_mols=df_mol
            )

            attempts += 1

            # clean up memory
            del mols
            del ligand_encoding
            del receptor_encoding
            del rand_rots
            torch.cuda.empty_cache()

        if len(valid_mols) == 0:
            fabric.print(f"No valid molecules sampled after {attempts} attempts.")

        seq_gen = save_samples(valid_mols, target_dirname, receptor, ligand_gt, config, fabric, seq_seed=seq_seed_h3)

        # compute metrics
        if use_metrics:
            try:
                evaluate_target_metrics(metrics_full, config, target_dirname=target_dirname, ligand_gt=ligand_gt, receptor=receptor, seq_gen=seq_gen, seq_seed=seq_seed_h3, attempts=attempts, receptor_id=receptor_id, t0=t0, fabric=fabric, df_mol=df_mol)
            except Exception as e:
                fabric.print(f"Error during metrics computation: {e}")
                fabric.print(f"Skipping receptor {receptor['id'][0]} #{receptor_id}")
        n_targets += 1
        if n_targets == config["sampling"]["n_targets"]:
            break
    if use_metrics:
        try:
            metrics_full.save()
            metrics_sampling = log_results(metrics_full, t0, config["sampling"]["docking_mode"], fabric=fabric)
        except Exception as e:
            fabric.print(f"Error during metrics computation: {e}")
            fabric.print(f"Skipping receptor {receptor['id'][0]} #{receptor_id}")

    return metrics_sampling


def sample_population(
    model,
    sampler,
    dec_module,
    config,
    fabric,
    ligand_encoding,
    receptor_encoding,
    rand_rots,
    label_batch=None,
    attempts=0,
    n_attempts=1,
    target_dirname=None
):
    """
    Samples a population of molecules
    """
    def score_fn(y: Tensor, sigma: float):
        B = y.shape[0]
        sigma = sigma * torch.ones(B, device=y.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        xhat = model(ligand_encoding=y, sigma=sigma, receptor_encoding=receptor_encoding, classes=label_batch)
        score = (xhat - y) / (sigma**2)
        return score
    with torch.no_grad():
        samples = sampler.sample(score_fn, y_init=ligand_encoding)
    if ("mcmc" in config["sampler"] and config["sampler"]["mcmc"]["save_trajectory"]) or ("save_trajectory" in config["sampler"] and config["sampler"]["save_trajectory"]):
        zhat = samples["xhat_traj"].squeeze(1)
    else:
        zhat = samples["sample"]

    # Render samples
    config_infer_ = copy.deepcopy(config)
    scale_factor = (128 / config_infer_["dset"]["grid_dim"]) ** 3
    if scale_factor > 0.0 and scale_factor != 1.0:
        config_infer_["sampling"]["batch_size_render"] = int(config_infer_["sampling"]["batch_size_render"] // scale_factor)
        config_infer_["sampling"]["batch_size_render_codes"] = max(1, int(config_infer_["sampling"]["batch_size_render_codes"] * scale_factor))
    mols = render_samples(
        zhat,
        dec_module,
        fabric,
        config_infer_,
        rand_rots=rand_rots,
        batch_size_render_codes=config_infer_["sampling"]["batch_size_render_codes"],
        target_dirname=target_dirname,
    )
    if len(mols) == 0:
        fabric.print(f"No molecules sampled at attempt {attempts} / {n_attempts}.")
    return mols


def compute_code_stats(
    loader_field,
    enc,
    config_nf,
    split,
    fabric,
    normalize_codes=True,
    field_maker=None,
    save_dir=None,
    debug=False,
):
    path_code_stats = os.path.join(config_nf["dirname"], f"codes_stats_{split}.pt")
    if os.path.exists(path_code_stats) and debug:
        fabric.print(f">> loading code stats from {path_code_stats}")
        code_stats = torch.load(path_code_stats, weights_only=False)
        code_stats = {
            k: v.to(fabric.device) if isinstance(v, torch.Tensor) else v
            for k, v in code_stats.items()
        }
        return code_stats

    # Determine number of samples per process
    if debug:
        n_samples_target = 10
    else:
        n_samples_target = 10_000

    # On-the-fly statistics computation (per process)
    enc.eval()
    n_samples = 0

    sum_codes_global = None
    sum_sq_codes_global = None
    max_codes_global = None
    min_codes_global = None
    median_abs_list = []
    total_spatial_elements = 0  # Track total number of spatial elements processed

    n = 2 * sum(config_nf["encoder"].get("downsample_map", [False, False, False]))
    subsampled_grid_dim = config_nf["dset"]["latent_grid_dim"] // n if n > 0 else config_nf["dset"]["latent_grid_dim"]

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader_field)):
            if n_samples >= n_samples_target:
                break

            codes = infer_codes_batch(
                batch, enc, field_maker, config_nf,
                code_stats=None, to_cpu=False, save_log_var=False
            )[0]

            if codes.size(0) == 0:
                continue

            # Gather codes from all processes
            codes = fabric.all_gather(codes)
            codes = codes.view(-1, config_nf["decoder"]["code_dim"], subsampled_grid_dim, subsampled_grid_dim, subsampled_grid_dim)

            remaining_samples = n_samples_target - n_samples
            if remaining_samples > 0:
                codes = codes[:remaining_samples]

            # Initialize on first batch
            if sum_codes_global is None:
                n_channels = codes.size(1)
                sum_codes_global = torch.zeros(n_channels, device=codes.device)
                sum_sq_codes_global = torch.zeros(n_channels, device=codes.device)
                max_codes_global = torch.full((n_channels,), float('-inf'), device=codes.device)
                min_codes_global = torch.full((n_channels,), float('inf'), device=codes.device)

            # Process each sample to match original get_stats computation
            for sample_idx in range(codes.size(0)):
                if n_samples >= n_samples_target:
                    break

                sample = codes[sample_idx]  # [C, D, D, D]

                # Compute global statistics per channel (like get_stats does)
                # This matches codes.mean((0,2,3,4)) - mean across batch and spatial dims
                for c in range(sample.size(0)):
                    channel_data = sample[c]  # [D, D, D]
                    channel_flat = channel_data.reshape(-1)  # [D*D*D]

                    # Update global statistics
                    sum_codes_global[c] += channel_flat.sum()
                    sum_sq_codes_global[c] += (channel_flat ** 2).sum()
                    max_codes_global[c] = max(max_codes_global[c], channel_flat.max())
                    min_codes_global[c] = min(min_codes_global[c], channel_flat.min())

                    # Collect median abs per channel per sample (for median computation)
                    median_abs_val = torch.median(torch.abs(channel_flat)).item()
                    if len(median_abs_list) <= c:
                        median_abs_list.append([])
                    median_abs_list[c].append(median_abs_val)

                n_samples += 1
                total_spatial_elements += subsampled_grid_dim ** 3

            # Clear codes tensor to free memory
            del codes
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if n_samples >= n_samples_target:
                break

    # Compute final statistics exactly like get_stats
    # Total elements per channel = n_samples * spatial_elements_per_sample
    total_elements_per_channel = n_samples * (subsampled_grid_dim ** 3)

    mean = sum_codes_global / total_elements_per_channel
    var = (sum_sq_codes_global / total_elements_per_channel) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    median_abs = np.array([np.median(channel_vals) for channel_vals in median_abs_list])

    # Clear intermediate tensors
    del sum_codes_global, sum_sq_codes_global, var

    # Format to match get_stats output - need to match the actual spatial dimensions of codes
    # get_stats uses amax((0,2,3,4), keepdim=True) which gives shape [1, C, 1, 1, 1]
    # But we need to broadcast to match the actual code dimensions [1, C, D, D, D]
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    max_codes_global = max_codes_global.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    min_codes_global = min_codes_global.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Print statistics using the same format as get_stats (only on rank 0)
    fabric.print(f"====codes {split}====")
    fabric.print(f"avg median of abs over all channels: {np.mean(median_abs)}")
    fabric.print(f"avg min over all channels: {min_codes_global.mean().item()}")
    fabric.print(f"avg max over all channels: {max_codes_global.mean().item()}")
    fabric.print(f"avg mean over all channels: {mean.mean().item()}")
    fabric.print(f"avg std over all channels: {std.mean().item()}")

    code_stats = {"mean": mean, "std": std}

    if normalize_codes:
        # Compute normalized stats using in-place operations where possible
        norm_max = max_codes_global.sub(mean).div_(std)
        norm_min = min_codes_global.sub(mean).div_(std)
        norm_median_abs = median_abs / std.mean(dim=(1,2,3,4)).cpu().numpy()

        fabric.print(f"====normalized codes {split}====")
        fabric.print(f"avg median of abs over all channels: {np.mean(norm_median_abs)}")
        fabric.print(f"avg min over all channels: {norm_min.mean().item()}")
        fabric.print(f"avg max over all channels: {norm_max.mean().item()}")
        fabric.print(f"avg mean over all channels: 0.0")
        fabric.print(f"avg std over all channels: 1.0")

        # Add plot_hist functionality like in original get_stats
        if save_dir is not None:
            plt.figure()
            plt.hist(median_abs, bins=30, alpha=0.7)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of median of abs values along dim=1')
            plt.savefig(f'{save_dir}/median_abs_histogram.png')
            plt.close()  # Use close instead of show for non-interactive
            fabric.print("Done plotting histogram")

        code_stats.update({
            "max_normalized": norm_max,
            "min_normalized": norm_min,
            "median_abs_normalized": norm_median_abs
        })
    else:
        code_stats.update({
            "max_normalized": max_codes_global,
            "min_normalized": min_codes_global,
            "median_abs_normalized": median_abs
        })

    # Clear median_abs_list to free memory
    del median_abs_list, median_abs

    # Move to device (already on device after all_gather)
    code_stats = {k: v.to(fabric.device) if isinstance(v, torch.Tensor) else v for k, v in code_stats.items()}

    if debug and fabric.global_rank == 0:
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in code_stats.items()}, path_code_stats)

    return code_stats


def plot_metric_vs_sigma(
    sigmas,
    metric_values,
    metric_name,
    y_label="Value",
    color="blue",
    marker="o",
    linestyle="-",
    save_dir=None,
    close_figure=True,
):
    """
    Generates and optionally saves a single plot of a metric vs. sigma.

    Args:
        sigmas (array-like): The sigma values (x-axis).
        metric_values (array-like): The metric values corresponding to sigmas (y-axis).
        metric_name (str): The name of the metric (used in title and filename).
        y_label (str, optional): Label for the y-axis. Defaults to "Value".
        color (str, optional): Plot line color. Defaults to 'blue'.
        marker (str, optional): Plot marker style. Defaults to 'o'.
        linestyle (str, optional): Plot line style. Defaults to '-'.
        save_dir (str, optional): Directory to save the plot. If None, plot is not saved.
                                   Defaults to None.
        close_figure (bool, optional): If True, closes the figure after saving (or generating).
                                       Useful for saving many plots without displaying.
                                       Defaults to False.
    """
    fig = plt.figure(figsize=(8, 6))  # Create a new figure
    plt.plot(
        sigmas,
        metric_values,
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=metric_name,
    )

    # Configuration
    plt.xscale("log")
    plt.xlabel("σ (Sigma)")
    plt.ylabel(y_label)
    plt.title(f"{metric_name} vs. σ")
    plt.xticks(sigmas, [f"{sigma:.2f}" for sigma in sigmas], rotation=45, ha="right")
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Saving the plot
    if save_dir:
        # Sanitize metric_name for use in filename (replace spaces, slashes etc.)
        safe_filename = "".join(c if c.isalnum() else "_" for c in metric_name)
        filename = os.path.join(save_dir, f"{safe_filename}_vs_sigma.png")
        try:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"Saved plot: {filename}")
        except Exception as e:
            print(f"Error saving plot {filename}: {e}")

    # Close the figure if requested (useful for batch saving)
    if close_figure:
        plt.close(fig)


def get_label(batch, config, num_classes=3):
    is_modality_cond = config["sampler"].get("modality_cond", False)
    if is_modality_cond:
        data_type = batch["ligand"]["data_type"] if batch["ligand"] is not None else batch["receptor"]["data_type"]
        batch_size = len(data_type)
        one_hot_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32)
        row_indices = torch.arange(batch_size, dtype=torch.long)
        one_hot_labels[row_indices, data_type] = 1.0
        return one_hot_labels.to(batch["receptor"]["atoms_channel"].device)
    return None


if __name__ == "__main__":
    main()
