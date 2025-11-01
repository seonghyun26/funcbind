from funcbind.dataset.field_maker import FieldMaker
from funcbind.models.adamw import AdamW
from funcbind.models.phema import PowerFunctionEMA
from funcbind.utils.utils_base import overwrite_config
import torch
import os
import copy
import numpy as np
from funcbind.utils.utils_nf import load_neural_field
from funcbind.models.denoiser import FuncBind
from collections import OrderedDict
from torch import nn
from omegaconf import OmegaConf


def create_funcbind(config: dict, code_stats: dict, fabric: object, num_classes=None):
    """
    Create a model based on the given configuration and decoder parameters.

    Args:
        config (dict): Configuration parameters for the model.

    Returns:
        model: The created model.
    """
    model = FuncBind(config, code_stats=code_stats, fabric=fabric, num_classes=num_classes)

    # n params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f">> FuncBind has {(n_params/1e6):.02f}M parameters")

    return model


def load_funcbind(
    pretrained_path: str,
    fabric = None,
    config = None,
    num_classes=None,
    train=False,
):
    """
    Loads a checkpoint file and restores the model and optimizer states.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        pretrained_path (str): The path to the directory containing the checkpoint file.
        optimizer (bool, optional): Whether to create an optimizer. Defaults to False.
        best_model (bool, optional): Whether to load the best model checkpoint or the regular checkpoint.
            Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model, optimizer (if provided), and the number of epochs trained.
    """
    checkpoint = fabric.load(os.path.join(pretrained_path, "checkpoint.pth.tar"))
    if config is None:
        config = checkpoint["config"]

    # code stats
    code_stats = checkpoint["code_stats"]
    for key in code_stats.keys():
        if isinstance(code_stats[key], np.ndarray):
            code_stats[key] = torch.tensor(code_stats[key])
        code_stats[key] = code_stats[key].to(fabric.device)

    # network
    model = create_funcbind(config, code_stats=code_stats, fabric=fabric, num_classes=num_classes)
    sd = "state_dict_ema" if not train else "state_dict"
    load_ema = True
    try:
        load_unet(checkpoint, model, fabric, sd=sd)
    except KeyError:
        fabric.print(">> no state_dict found, loading state_dict_ema")
        load_ema = False  # EMA and model have the same weights
        load_unet(checkpoint, model, fabric, sd="state_dict_ema")
    fabric.print(f">> loaded model trained for {checkpoint['epoch']} epochs")

    # EMA
    if train:
        with torch.no_grad():
            assert "ema_stds" in config and config["ema_stds"] is not None and (len(config["ema_stds"]) > 0 and config["ema_stds"][0] != 0.0), "ema_stds must be set and non-empty"
            fabric.print(">> using PowerFunctionEMA with stds", config["ema_stds"])
            model_ema = PowerFunctionEMA(model, stds=config["ema_stds"])
            if load_ema:
                load_unet(checkpoint, model_ema.emas[0], fabric, sd="state_dict_ema")
            fabric.print(">> loaded model_ema")

    acc_iter = checkpoint.get("acc_iter", 0)

    if train:
        return model, model_ema, checkpoint["optimizer"], code_stats, acc_iter
    else:
        return model, code_stats, acc_iter


def learning_rate_schedule(optimizer, iteration, config, world_size=1):
    if "use_lr_schedule" not in config or ("use_lr_schedule" in config and not config["use_lr_schedule"]):
        return config["lr"]
    ref_lr = config["lr"]
    ref_batches = config["ref_batches"]
    batch_size = config["dset"]["batch_size"] * world_size
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(iteration / (ref_batches * batch_size), 1))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def load_funcbind_sampling(config, fabric):
    checkpoint_fb = fabric.load(os.path.join(config["fb_pretrained_path"], "checkpoint.pth.tar"))
    if "sampler" in checkpoint_fb["config"]:
        checkpoint_fb["config"]["sampler"] = {}
        if "mcmc" in checkpoint_fb["config"]["sampler"]:
            checkpoint_fb["config"]["sampler"]["mcmc"] = {}
    config = overwrite_config(checkpoint_fb["config"], config)

    config["dset"]["data_aug"] = False
    fabric.print(f"updated config:\n{OmegaConf.to_yaml(config)}")

    num_classes = num_classes_funcbind(config)

    # load checkpoint
    with torch.no_grad():
        fabric.print(">> loading nf checkpoint from", {config["nf_pretrained_path"]})
        nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
        config_nf = nf_checkpoint["config"]
        config_nf["dset"]["use_single_dataset"] = config["dset"].get("use_single_dataset", None)

        # remove for the newer checkpoints
        config["dset"]["grid_dim"] = config_nf["dset"]["grid_dim"]
        config["dset"]["resolution"] = config_nf["dset"]["resolution"]
        config["encoder"] = config_nf["encoder"]
        config["decoder"] = config_nf["decoder"]

        enc, dec = load_neural_field(nf_checkpoint, fabric, config_nf)

        fabric.print(">> loading fb checkpoint from", config["fb_pretrained_path"])
        model, code_stats, _ = load_funcbind(config["fb_pretrained_path"], fabric=fabric, config=config, num_classes=num_classes)
        model = fabric.setup_module(model)

        dec_module = dec.module if hasattr(dec, "module") else dec
        dec_module.set_code_stats(code_stats)

    return model, enc, dec_module, config, config_nf, num_classes


def load_unet(
    checkpoint: dict,
    net: nn.Module,
    fabric: object,
    sd: str = None,
) -> nn.Module:
    """
    Load a neural network's state dictionary from a checkpoint and update the network's parameters.

    Args:
        checkpoint (dict): A dictionary containing the checkpoint data.
        net (nn.Module): The neural network model to load the state dictionary into.
        fabric (object): An object with a print method for logging.
        net_name (str, optional): The key name for the network's state dictionary in the checkpoint. Defaults to "dec".
        is_compile (bool, optional): A flag indicating whether the network is compiled. Defaults to True.
        sd (str, optional): A specific key for the state dictionary in the checkpoint. If None, defaults to using net_name.

    Returns:
        nn.Module: The neural network model with the loaded state dictionary.
    """
    net_dict = net.state_dict()
    weight_first_layer_before = next(iter(net_dict.values())).sum()
    new_state_dict = OrderedDict()
    if 'emas' in checkpoint[sd]:  # for PowerFunctionEMA
        state_dicts_ckpt = checkpoint[sd]['emas'][0]
    else:
        state_dicts_ckpt = checkpoint[sd]
    for k, v in state_dicts_ckpt.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    weight_first_layer_after = next(iter(net_dict.values())).sum()
    assert weight_first_layer_before != weight_first_layer_after, "loading did not work"
    fabric.print(">> loaded denoiser")

    return net


def create_optimizer(funcbind, config, fabric):
    if config["wd"] >= 0:
        fabric.print(f">> using AdamW with weight decay {config['wd']}")
        optimizer = AdamW(
            funcbind.parameters(),
            lr=config["lr"],
            weight_decay=config["wd"],
            betas=(0.9, config["dset"]["beta2"]),
        )
    else:
        fabric.print(">> using Adam")
        optimizer = torch.optim.Adam(
            funcbind.parameters(), lr=config["lr"], betas=(0.9, config["dset"]["beta2"])
        )
    optimizer.zero_grad()
    return optimizer


def num_classes_funcbind(config):
    if "modality_cond" in config["sampler"] and config["sampler"]["modality_cond"]:
        num_classes = 3
    else:
        num_classes = 0
    return num_classes


def create_field_makers(config, config_nf, fabric):
    field_maker = FieldMaker(config_nf, sample_points=False)
    field_maker = field_maker.to(fabric.device)

    # Determine downsample map based on denoiser configuration
    downsample_map = config["denoiser"].get("downsample_map", [False, False])

    # Calculate downsampling factor based on encoder configuration
    encoder_has_downsampling = sum(config_nf["encoder"].get("downsample_map", [False, False, False])) > 0
    downsampling_factor = max(1, sum(downsample_map) * (1 if encoder_has_downsampling else 2))

    if downsampling_factor > 0:
        config_receptor = copy.deepcopy(config_nf)
        config_receptor["dset"]["latent_grid_dim"] = (config_nf["dset"]["latent_grid_dim"] * downsampling_factor)
        config_receptor["dset"]["latent_resolution"] = (config_nf["dset"]["latent_resolution"] / downsampling_factor)
        field_maker_receptor = FieldMaker(config_receptor, sample_points=False).to(fabric.device)
    else:
        field_maker_receptor = FieldMaker(config_nf, sample_points=False).to(fabric.device)
    return field_maker, field_maker_receptor