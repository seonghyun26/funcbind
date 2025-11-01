# this is an adaptation of EDM2
# https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py

import numpy as np
import torch


#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.
def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.
_constant_cache = dict()

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format
    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def resample(x, f=[1, 1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    # Ensure the filter is a 1D array with an odd length
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    # Create a separable 3D filter by computing the outer product
    f3d = f[:, None, None] * f[None, :, None] * f[None, None, :]
    f3d = f3d[np.newaxis, np.newaxis, :, :, :]  # Shape: (1, 1, D, H, W)
    # f3d = torch.Tensor(f3d, dtype=x.dtype, device=x.device)
    f3d = torch.Tensor(f3d).detach().clone().to(x.device).type(x.dtype)
    c = x.shape[1]  # Number of channels
    if mode == 'down':
        # Downsample: apply convolution with stride 2
        return torch.nn.functional.conv3d(
            x,
            f3d.repeat(c, 1, 1, 1, 1),
            groups=c,
            stride=2,
            padding=pad
        )
    elif mode == 'up':
        # Upsample: apply transposed convolution with stride 2 and scale the filter
        return torch.nn.functional.conv_transpose3d(
            x,
            (f3d * 8).repeat(c, 1, 1, 1, 1),
            groups=c,
            stride=2,
            padding=pad
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes are 'keep', 'down', and 'up'.")

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).
def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).
def mp_sum(a, b, t=0.5):
    assert a.shape == b.shape, f"{a.shape=}, {b.shape=}"
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).
def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t

    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).
class MPConv(torch.nn.Module):
    def __init__(self, in_inp_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_inp_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)  # i wonder why this is necessary
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 5
        return torch.nn.functional.conv3d(x, w, padding=(w.shape[-1]//2,))


class BlockUncond(torch.nn.Module):
    def __init__(self,
        in_inp_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_inp_channels, out_channels, kernel=[3,3,3])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3,3])
        self.conv_skip = MPConv(in_inp_channels, out_channels, kernel=[1,1,1]) if in_inp_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1,1]) if self.num_heads != 0 else None

    def forward(self, x):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3] * y.shape[4])
            q, k, v = normalize(y, dim=2).unbind(3) # voxel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)


        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).
class Block(torch.nn.Module):
    def __init__(self,
        in_inp_channels,                # Number of input channels.
        out_channels,                   # Number of output channels.
        cond_channels,                  # Number of condtiional channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.cond_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_inp_channels, out_channels, kernel=[3,3,3])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3,3])
        self.cond_linear = MPConv(cond_channels, out_channels, kernel=[1,1,1])
        self.conv_skip = MPConv(in_inp_channels, out_channels, kernel=[1,1,1]) if in_inp_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1,1]) if self.num_heads != 0 else None

    def forward(self, x, x_cond):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))

        # downsample x_cond to be same dimensions as x and transform to same channels as y
        resample_ratio = x_cond.shape[-1] // x.shape[-1]
        if resample_ratio > 1:
            x_cond = torch.nn.functional.avg_pool3d(x_cond, kernel_size=resample_ratio, stride=resample_ratio)

        # Aggregate conditional information.
        x_cond = self.cond_linear(x_cond, gain=self.cond_gain) + 1
        y = mp_silu(y * x_cond.to(y.dtype))
        # x-attn  # TODO

        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3] * y.shape[4])
            q, k, v = normalize(y, dim=2).unbind(3) # voxel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).
class UNet3DCondNoise(torch.nn.Module):
    def __init__(self,
        code_grid_dim,                    # spatial-z code grid dim.
        n_inp_channels,                   # n channels of spatial-z code.
        model_channels      = 64,         # Base multiplier for the number of channels.
        ch_mults        = [1,2,3,4],      # Per-resolution multipliers for the number of channels.
        n_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [4,2],      # List of resolutions with self-attention.
        concat_balance      = 0.5,        # Balance between skip connections (0) and main path (1).
        class_dim = 0,
        **block_kwargs,                   # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in ch_mults]
        cnoise = cblock[0]
        ccond = n_inp_channels  # cblock[0]  # max(cblock)  # TODO: ccond need to be the same dim as the receptor encoding
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.concat_balance = concat_balance
        self.act = mp_silu

        if code_grid_dim != 16:
            print("If code_grid_dim != 16, remember to change attn_resolutions")

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, ccond, kernel=[])
        self.emb_length = MPConv(class_dim, ccond, kernel=[]) if class_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = n_inp_channels + 1
        for level, channels in enumerate(cblock):
            res = code_grid_dim >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, ccond, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(n_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, ccond, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = code_grid_dim >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, ccond, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, ccond, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, ccond, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(n_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, ccond, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, n_inp_channels, kernel=[3,3,3])

    def forward(self, ligand, receptor, noise, classes, cfg_dropout=0.0):
        # Embedding of noise.
        emb = self.emb_noise(self.emb_fourier(noise))

        # CFG: randomly make cfg_dropout% of batch unconditional
        cfg_mask = None
        if cfg_dropout > 0:
            cfg_mask = torch.rand(ligand.shape[0], device=ligand.device) < cfg_dropout

        # Class conditioning (skip for cfg_mask elements)
        if self.emb_length is not None:
            class_emb = self.emb_length(classes * np.sqrt(classes.shape[1]))
            if class_emb.shape[0] == 1:
                class_emb = class_emb.repeat(emb.shape[0], 1)
            if cfg_mask is not None:
                emb[~cfg_mask] = mp_sum(emb[~cfg_mask], class_emb[~cfg_mask], t=0.5)
            else:
                emb = mp_sum(emb, class_emb, t=0.5)
        # emb = mp_silu(emb)  # for old checkpoints - dimension 128x1x1x1

        # Concat noise embedding and receptor
        emb = emb[..., None, None, None]               # (B,C,1,1,1)
        emb = emb.expand(-1, -1, *receptor.shape[2:])  # (B,C,D,H,W)
        if cfg_mask is not None:
            emb = torch.where(
                cfg_mask[:, None, None, None, None],
                emb,  # Unconditional: keep just noise embedding
                mp_sum(receptor, emb, t=0.5)  # Conditional: combine receptor + noise
            )
        else:
            emb = mp_sum(receptor, emb, t=0.5)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([ligand, torch.ones_like(ligand[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x


if __name__ == "__main__":
    # Example usage
    n_inp_channels = 576
    n_receptor_channels = 4
    grid_res = 16
    batch_sz = 32
    noise = torch.randn([batch_sz])
    unet = UNet3DCondNoise(code_grid_dim=grid_res, n_inp_channels=n_inp_channels)
    ligand_tensor = torch.randn(batch_sz, n_inp_channels, grid_res, grid_res, grid_res)
    receptor_tensor = torch.randn(batch_sz, n_inp_channels, grid_res, grid_res, grid_res)
    classes = None  # or torch.randn(batch_sz, class_dim) if using classes

    output_tensor = unet(ligand_tensor, receptor_tensor, noise, classes)
    print(ligand_tensor.shape, output_tensor.shape)
