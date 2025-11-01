import hydra
import matplotlib.pyplot as plt
import numpy as np
from funcbind.distributions.distributions import ClippedLogNormalSigma, UniformMeasurement
from funcbind.utils.utils_sampling import normalize_code
import torch
import matplotlib.pyplot as plt

from funcbind.utils.utils_nf import create_receptor_encoder
from funcbind.utils.constants import N_RECEPTOR_ELEMENTS
from funcbind.models.unet3d import UNet3DCondNoise, MPFourier, MPConv

import hydra
from funcbind.utils.utils_base import setup_fabric


class FuncBind(torch.nn.Module):
    def __init__(self, config, code_stats: dict = None, fabric = None, num_classes: int = None):
        super().__init__()
        self.code_stats = code_stats
        self.code_dim = config["decoder"]["code_dim"]
        n = 2 * sum(config["encoder"].get("downsample_map", [False, False, False]))
        self.code_grid_dim = config["dset"]["latent_grid_dim"] // n if n > 0 else config["dset"]["latent_grid_dim"]
        self.device = fabric.device
        self.n_channels_receptor = N_RECEPTOR_ELEMENTS
        if config["sampler"]["_target_"] == "funcbind.sampling.DiffusionSampler":
            fabric.print(f"Using ClippedLogNormalSigma for log_sigma_mean={config['sampler']['log_sigma_mean']}, log_sigma_std={config['sampler']['log_sigma_std']}, sigma_max=100.0")
            self.sigma_distribution = ClippedLogNormalSigma(log_sigma_mean=config["sampler"]["log_sigma_mean"], log_sigma_std=config["sampler"]["log_sigma_std"], sigma_max=100.0)
        elif config["sampler"]["_target_"] == "funcbind.sampling.MultiMeasurementOATSampler":
            fabric.print(f"Using UniformMeasurement for sigma={config['sampler']['sigma']}, m={config['sampler']['m']}")
            self.sigma_distribution = UniformMeasurement(sigma=config["sampler"]["sigma"], m=config["sampler"]["m"])
        elif config["sampler"]["_target_"] == "funcbind.sampling.SingleMeasurementSampler":
            fabric.print(f"Using UniformMeasurement for sigma={config['sampler']['sigma']}")
            self.sigma_distribution = UniformMeasurement(sigma=config["sampler"]["sigma"], m=1)

        self.class_dim = num_classes if (config["sampler"].get("modality_cond", False)) else 0
        self.receptor_encoder = create_receptor_encoder(config, self.n_channels_receptor)

        self.cfg_dropout = config["denoiser"]["cfg_dropout"] if "cfg_dropout" in config["denoiser"] else 0.0
        if self.cfg_dropout > 0:
            fabric.print(f"Using CFG dropout: {self.cfg_dropout}")

        fabric.print(">> Loading UNet3DCondNoise model")
        self.unet3d = UNet3DCondNoise(
            code_grid_dim=self.code_grid_dim,
            n_inp_channels=config["decoder"]["code_dim"],
            model_channels=config["denoiser"]["model_channels"],
            ch_mults=config["denoiser"]["ch_mults"],
            n_blocks=config["denoiser"]["n_blocks"],
            attn_resolutions=config["denoiser"]["attn_resolutions"],
            dropout=config["denoiser"]["dropout"],
            class_dim=self.class_dim
        )

        # preconditioning attributes
        # https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py#L285
        self.sigma_data = 1.  # since we normalize the codes
        logvar_channels = 128 # maybe change this later?
        if config["sampler"]["_target_"] != "funcbind.sampling.SingleMeasurementSampler":
            self.logvar_fourier = MPFourier(logvar_channels)
            self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(
        self,
        ligand_encoding: torch.Tensor,
        sigma: torch.Tensor,
        receptor: torch.Tensor = None,
        receptor_encoding: torch.Tensor = None,
        return_logvar=False,
        preconditioning=True,
        classes=None,
        cfg_dropout=True
    ) -> torch.Tensor:
        assert receptor_encoding is not None or receptor is not None, "Either receptor_encoding or receptor must be provided"
        if receptor_encoding is None:
            receptor_encoding = self.receptor_encoder(receptor) if self.receptor_encoder is not None else receptor

        classes = None if self.class_dim == 0 else torch.zeros([1, self.class_dim], device=self.device) if classes is None else classes.to(torch.float32).reshape(-1, self.class_dim)

        if preconditioning:
            # preconditioning
            sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1, 1)
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = sigma.flatten().log() / 4

            # forward unet
            x_in = (c_in * ligand_encoding)
            F_x = self.unet3d(x_in, receptor_encoding, c_noise, classes=classes, cfg_dropout=self.cfg_dropout if cfg_dropout else 0.0)
            D_x = c_skip * ligand_encoding + c_out * F_x
        else:
            D_x = self.unet3d(ligand_encoding, receptor_encoding, sigma, classes=classes, cfg_dropout=self.cfg_dropout if cfg_dropout else 0.0)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x


    def score(
        self,
        y: torch.Tensor,
        sigma: torch.Tensor,
        receptor: torch.Tensor = None,
        receptor_encoding: torch.Tensor = None,
    ) -> torch.Tensor:
        assert receptor_encoding is not None or receptor is not None, "Either receptor_encoding or receptor must be provided"
        if receptor_encoding is None:
            receptor_encoding = self.receptor_encoder(receptor) if self.receptor_encoder is not None else receptor

        xhat = self.forward(y, receptor_encoding=receptor_encoding, sigma=sigma)
        sigma = sigma.view(-1, 1, 1, 1, 1)
        score = (xhat - y) / (sigma ** 2)
        return score


########################################################################################
def unsqueeze_trailing(x, n):
    """
    adds n trailing singleton dimensions to x
    """
    return x.reshape(*x.shape, *((1,) * n))


def add_noise_to_code(codes: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to the input codes.

    Args:
        codes (torch.Tensor): Input codes to which noise will be added.

    Returns:
        torch.Tensor: Codes with added noise.
        torch.Tensor: Noise added to the codes.
    """
    # x [B, ...]
    # sigma [B]
    if isinstance(sigma, float) or isinstance(sigma, int):
        sigma = float(sigma) * torch.ones(codes.shape[0], device=codes.device, dtype=codes.dtype)

    # [B, ...]
    sigma = unsqueeze_trailing(sigma, codes.ndim - 1)

    return codes + sigma * torch.randn_like(codes)


def process_codes(
    codes: torch.Tensor,
    fabric: object,
    split: str,
    normalize_codes: bool,
    plot_hist = False,
    save_dir = None
) -> dict:
    """
    Process the codes from the checkpoint.

    Args:
        checkpoint (dict): The checkpoint containing the codes.
        logger (object): The logger object for logging messages.
        device (torch.device): The device to use for processing the codes.
        is_filter (bool, optional): Whether to filter the codes. Defaults to False.

    Returns:
        tuple: A tuple containing the processed codes, statistics, and normalized codes.
    """
    max, min, mean, std, median_abs = get_stats(
        codes,
        fabric=fabric,
        message=f"====codes {split}====",
    )
    code_stats = {
        "mean": mean,
        "std": std,
    }
    if normalize_codes:
        codes = normalize_code(codes, code_stats)
        max_normalized, min_normalized, _, _, median_abs_normalized = get_stats(
            codes,
            fabric=fabric,
            message=f"====normalized codes {split}====",
            plot_hist=plot_hist,
            save_dir = save_dir
        )
    else:
        max_normalized, min_normalized, median_abs_normalized = max, min, median_abs
    code_stats.update({
        "max_normalized": max_normalized,
        "min_normalized": min_normalized,
        "median_abs_normalized": median_abs_normalized
    })
    # clean memory
    del codes
    del max
    del min
    del mean
    del std
    del median_abs
    del max_normalized
    del min_normalized
    del median_abs_normalized
    torch.cuda.empty_cache()
    return code_stats


def get_stats(
    codes: torch.Tensor,
    fabric: object = None,
    message: str = None,
    plot_hist: bool = False,
    save_dir = None
):
    """
    Calculate statistics of the input codes.

    Args:
        codes_init (torch.Tensor): The input codes.
        fabric (object, optional): The logger object for logging messages. Defaults to None.
        message (str, optional): Additional message to log. Defaults to None.

    Returns:
        tuple: A tuple containing the calculated statistics:
            - max (torch.Tensor): The maximum values with shape [1, C, 1, 1, 1].
            - min (torch.Tensor): The minimum values with shape [1, C, 1, 1, 1].
            - mean (torch.Tensor): The mean values with shape [1, C, 1, 1, 1].
            - std (torch.Tensor): The standard deviation values with shape [1, C, 1, 1, 1].
            - median_abs (np.array): Median absolute values per channel.
    """
    if message is not None:
        fabric.print(message)
    median_abs = np.array([torch.median(torch.abs(codes[:, c]).reshape(-1).cpu()).item() for c in range(codes.size(1))])
    max_ = codes.amax((0,2,3,4), keepdim=True)
    min_ = codes.amin((0,2,3,4), keepdim=True)
    mean = codes.mean((0,2,3,4), keepdim=True)
    std = codes.std((0,2,3,4), keepdim=True)

    fabric.print(f"avg median of abs over all channels: {np.mean(median_abs)}")
    fabric.print(f"avg min over all channels: {min_.mean().item()}")
    fabric.print(f"avg max over all channels: {max_.mean().item()}")
    fabric.print(f"avg mean over all channels: {mean.mean().item()}")
    fabric.print(f"avg std over all channels: {std.mean().item()}")
    fabric.print(f"codes size: {codes.shape}")
    fabric.print(f"stats shapes: max={max_.shape}, min={min_.shape}, mean={mean.shape}, std={std.shape}")

    if plot_hist:
        plt.figure()
        plt.hist(median_abs, bins=30, alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of median of abs values along dim=1')
        plt.savefig(f'{save_dir}/median_abs_histogram.png')
        plt.show()

        values_along_dim1 = codes.view(codes.size(0), -1)
        values_np = values_along_dim1.cpu().numpy()
        all_values = values_np.flatten()
        plt.figure()
        plt.hist(all_values, bins=30, alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of all values')
        plt.savefig(f'{save_dir}/all_histogram.png')
        plt.show()

        fabric.print("Done plotting histogram")

    return max_, min_, mean, std, median_abs

def log_metrics(exp_name, epoch, train_loss, val_loss, sampling_metrics, best_res, time, fabric):
    """
    Logs the metrics for a given epoch.

    Args:
        epoch (int): The current epoch number.
        train_loss (float): The training loss value.
        val_loss (float): The validation loss value.
        sampling_metrics (dict): The dictionary containing additional sampling metrics.
        time (float): The time taken for the epoch.
        fabric: The logger object for writing the metrics.

    Returns:
        None
    """
    str_ = f">> {exp_name} epoch: {epoch} ({time:.2f}s)\n"
    str_ += f"[train_loss] {train_loss:.2f} |"
    if val_loss is not None and best_res is not None:
        str_ += f" | [val_loss] {val_loss:.2f} (best: {best_res:.2f})"
    if sampling_metrics is not None:
        str_ += "\n| [sampling_miou]"
        for k, v in sampling_metrics.items():
            str_ += f" | {k}: {v:.4f}"
    fabric.print(str_)


@hydra.main(config_path="../configs", config_name="train_fb", version_base=None)
def main(config):
    # Run python -m funcbind.models.denoiser --config-name=train_fb
    from funcbind.utils.utils_fb import num_classes_funcbind, create_funcbind
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    try:
        # This code will now work without errors
        config.decoder = OmegaConf.create()
        config.decoder.code_dim = 128
        config.encoder = OmegaConf.create()
        config.encoder.downsample_map = [False, False, False]
    finally:
        # Crucially, re-lock the config to restore its original behavior
        OmegaConf.set_struct(config, True)

    batch_sz = 128
    spatial_dim = 16
    fabric = setup_fabric(config)
    num_classes = num_classes_funcbind(config)
    model = create_funcbind(config, code_stats=None, fabric=fabric, num_classes=num_classes)
    model = fabric.setup_module(model)
    model.eval()

    ligand_encoding = torch.randn(batch_sz, config["decoder"]["code_dim"], spatial_dim, spatial_dim, spatial_dim).to(fabric.device)
    receptor = torch.randn(batch_sz, N_RECEPTOR_ELEMENTS, 2 * spatial_dim, 2 * spatial_dim, 2 * spatial_dim).to(fabric.device)
    sigma = torch.randn([batch_sz]).to(fabric.device)

    # xhat, logvar = model(ligand_encoding, sigma, receptor, return_logvar=True)
    xhat = model(ligand_encoding, sigma, receptor, return_logvar=False)
    print(f">> dimension z: {ligand_encoding.shape}, dimension xhat: {xhat.shape}")
    score = model.score(xhat, sigma, receptor)
    print(f">> dimension score: {score.shape}")


if __name__ == "__main__":
    main()
