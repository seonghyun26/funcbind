import einops
import torch


class CategoricalValue(torch.distributions.Distribution):
    def __init__(self, values: torch.Tensor, categorical: torch.distributions.Categorical):
        if values.shape[0] != categorical.probs.shape[0]:
            raise RuntimeError(f"{values.shape[0]=} != {categorical.probs.shape[0]=}")
        self.values = values
        self.categorical = categorical

    def sample(self, sample_shape=torch.Size([])):
        return self.values[self.categorical.sample(sample_shape)]

    @property
    def mean(self):
        return einops.einsum(self.values, self.categorical.probs, "i ..., i ... -> ...")

    def __repr__(self):
        return "CategoricalValue"


class WeightedMeasurement(CategoricalValue):
    def __init__(self, sigma: float, probs: torch.Tensor):
        self.sigma = sigma
        self.m = probs.shape[0]
        values = sigma * torch.arange(1, self.m + 1).pow(-0.5)
        super().__init__(values=values, categorical=torch.distributions.Categorical(probs=probs))

    def __repr__(self):
        return f"WeightedMeasurement(sigma={self.sigma}, logits={self.logits})"


class UniformMeasurement(WeightedMeasurement):
    def __init__(self, sigma: float, m: int):
        probs = torch.ones(m)
        super().__init__(sigma=sigma, probs=probs)

    def __repr__(self):
        return f"UniformMeasurement(sigma={self.sigma}, m={self.m})"


class Normal(torch.distributions.Distribution):
    def __init__(self, config, device=None, dtype=None):
        self.sigma = config["sampler"]["sigma"] if config["sampler"]["name"] != "diffusion" else config["sampler"]["sigma_max"]
        self.config = config
        self.device = device
        self.dtype = dtype

    def sample(self, sample_shape=torch.Size([])):
        if "latent_grid_dim" in self.config["dset"]:
            n = 2 * sum(self.config["encoder"].get("downsample_map", [False, False, False]))
            latent_grid_dim = self.config["dset"]["latent_grid_dim"] // n if n > 0 else self.config["dset"]["latent_grid_dim"]
        else:
            latent_grid_dim = 16
        shape = (
            self.config["sampling"]["n_chains"],
            self.config["decoder"]["code_dim"],
            latent_grid_dim,
            latent_grid_dim,
            latent_grid_dim,
        )
        return torch.randn_like(torch.zeros(shape, device=self.device, dtype=self.dtype)) * self.sigma

    def __repr__(self):
        return f"Normal(sigma={self.sigma}, sample_shape={self.config['sampling']['n_chains']})"


class UniformPlusNormal(torch.distributions.Distribution):
    def __init__(self, code_stats, config, device=None, dtype=None):
        super().__init__()
        self.sigma = config["sampler"]["sigma"] if config["sampler"]["name"] != "diffusion" else config["sampler"]["sigma_max"]
        self.code_stats = code_stats
        self.device = device
        self.dtype = dtype
        self.config = config

    def sample(self, sample_shape=torch.Size([])):
        # Initialize with zeros
        if "latent_grid_dim" in self.config["dset"]:
            n = 2 * sum(self.config["encoder"].get("downsample_map", [False, False, False]))
            latent_grid_dim = self.config["dset"]["latent_grid_dim"] // n if n > 0 else self.config["dset"]["latent_grid_dim"]
        else:
            latent_grid_dim = 16
        shape = (
            self.config["sampling"]["n_chains"],
            self.config["decoder"]["code_dim"],
            latent_grid_dim,
            latent_grid_dim,
            latent_grid_dim,
        )
        result = torch.zeros(shape, device=self.device, dtype=self.dtype)

        # Add uniform noise based on min/max values for each channel
        for i in range(shape[1]):  # Iterate over channels (code_dim)
            min_val = self.code_stats["min_normalized"][0, i].item()
            max_val = self.code_stats["max_normalized"][0, i].item()

            uniform_noise = torch.empty(
                (shape[0], *shape[2:]),
                device=self.device,
                dtype=self.dtype
            ).uniform_(min_val, max_val)

            result[:, i] = uniform_noise

        # Add Gaussian noise
        result = result + torch.randn_like(result) * self.sigma

        return result

    def __repr__(self):
        return f"UniformPlusNormal(sigma={self.sigma}, sample_shape={self.sample_shape})"


class ClippedLogNormalSigma(torch.distributions.Distribution):
    def __init__(self, log_sigma_mean: float, log_sigma_std: float, sigma_max: float = 100.0):
        self.log_sigma_mean = log_sigma_mean
        self.log_sigma_std = log_sigma_std
        self.log_sigma_dist = torch.distributions.Normal(log_sigma_mean, log_sigma_std)
        self.sigma_max = sigma_max

    def sample(self, sample_shape=torch.Size([])):
        log_sigma = self.log_sigma_dist.sample(sample_shape)
        sigma = log_sigma.exp()
        sigma = torch.clamp(sigma, max=self.sigma_max)
        return sigma

    def __repr__(self):
        return f"LogNormalSigma(log_sigma_mean={self.log_sigma_mean}, log_sigma_std={self.log_sigma_std}, sigma_max={self.sigma_max})"
