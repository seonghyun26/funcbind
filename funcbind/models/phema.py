# adapted from EDM2
# https://github.com/NVlabs/edm2/blob/main/training/phema.py

import copy
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from contextlib import contextmanager

#----------------------------------------------------------------------------
# Convert power function exponent to relative standard deviation
# according to Equation 123.

def exp_to_std(exp):
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std

#----------------------------------------------------------------------------
# Convert relative standard deviation to power function exponent
# according to Equation 126 and Algorithm 2.

def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp

#----------------------------------------------------------------------------
# Construct response functions for the given EMA profiles
# according to Equations 121 and 108.

def power_function_response(ofs, std, len, axis=0):
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp

#----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equation 151 and Algorithm 3.

def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den

#----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile during training
# according to Equation 127.

def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Class for tracking power function EMA during the training.

class PowerFunctionEMA(nn.Module):
    @torch.no_grad()
    def __init__(self, net, stds=[0.100], foreach=True):
        super(PowerFunctionEMA, self).__init__()
        self.net = net
        self.stds = stds
        self.foreach = foreach
        self.emas = [copy.deepcopy(net) for _std in stds]
        device = getattr(net, "device", next(net.parameters()).device)
        self.to(device)

    @staticmethod
    def _parameter_buckets(net, ema):
        buckets = {}
        for p_net, p_ema in zip(net.parameters(), ema.parameters()):
            key = (p_ema.device, p_ema.dtype)
            net_params, ema_params = buckets.setdefault(key, ([], []))
            net_params.append(p_net)
            ema_params.append(p_ema)
        return buckets.values()

    @torch.no_grad()
    def to(self, *args, **kwargs):
        for ema in self.emas:
            ema.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def reset(self):
        for ema in self.emas:
            if self.foreach:
                for net_params, ema_params in self._parameter_buckets(self.net, ema):
                    torch._foreach_copy_(ema_params, net_params)
            else:
                for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                    p_ema.copy_(p_net)

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size)
            if self.foreach:
                for net_params, ema_params in self._parameter_buckets(self.net, ema):
                    torch._foreach_lerp_(ema_params, net_params, 1 - beta)
            else:
                for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                    p_ema.lerp_(p_net, 1 - beta)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.buffers(), ema.buffers()):
                p_ema.copy_(p_net)
        return [(ema, f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def state_dict(self):
        return dict(stds=self.stds, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.stds = state['stds']
        for ema, s_ema in zip(self.emas, state['emas']):
            ema.load_state_dict(s_ema)


class CPUPowerFunctionEMA:
    """FP32 shadows on rank zero, without constructing another GPU model.

    Transfers for updates are bounded to 64 MiB. Evaluation borrows the original
    module in place and restores its weights and per-module train/eval flags even
    on failure. This temporarily needs another model-sized CPU backup, not VRAM.
    The serialized format remains compatible with PowerFunctionEMA/sampling.
    """
    def __init__(self, net, stds=(0.100,), chunk_numel=16 * 1024**2):
        self.net = net
        self.stds = list(stds)
        self.chunk_numel = max(1, int(chunk_numel))
        self.shadows = [OrderedDict(
            (name, value.detach().to(device="cpu", copy=True))
            for name, value in net.state_dict().items()
        ) for _ in stds]

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        for std, shadow in zip(self.stds, self.shadows):
            beta = power_function_beta(std, cur_nimg, batch_size)
            for name, parameter in self.net.named_parameters():
                if not parameter.is_contiguous():
                    raise ValueError(f"CPU EMA requires contiguous parameters: {name}")
                source, dest = parameter.detach().view(-1), shadow[name].view(-1)
                for start in range(0, source.numel(), self.chunk_numel):
                    end = start + self.chunk_numel
                    dest[start:end].lerp_(source[start:end].to("cpu"), 1 - beta)

    @torch.no_grad()
    def state_dict(self):
        for shadow in self.shadows:
            for name, value in self.net.named_buffers():
                shadow[name].copy_(value)
        return dict(stds=self.stds, emas=self.shadows)

    @torch.no_grad()
    def load_state_dict(self, state):
        if len(state["emas"]) != len(self.shadows):
            raise ValueError("EMA profile count differs from checkpoint")
        self.stds = list(state["stds"])
        for shadow, saved in zip(self.shadows, state["emas"]):
            # Base density-free checkpoints intentionally omit the new branch.
            for name, value in saved.items():
                name = name.removeprefix("module.")
                if name in shadow:
                    shadow[name].copy_(value)

    @contextmanager
    def average_parameters(self):
        state = self.net.state_dict()
        shadow = self.state_dict()["emas"][0]
        modes = [(module, module.training) for module in self.net.modules()]
        # Allocate the backup before changing any parameter so allocation failure
        # cannot leave a half-swapped training model.
        backup = OrderedDict((name, value.detach().to("cpu", copy=True))
                             for name, value in state.items())
        try:
            with torch.no_grad():
                for name, value in state.items():
                    value.copy_(shadow[name])
            self.net.eval()
            yield self.net
        finally:
            with torch.no_grad():
                for name, value in state.items():
                    value.copy_(backup[name])
            for module, training in modes:
                module.training = training


def create_ema(net, config, fabric):
    stds = config["ema_stds"]
    if not stds or not stds[0]:
        raise ValueError("ema_stds must be non-empty and nonzero")
    if config.get("performance", {}).get("ema_cpu", False):
        if config.get("precision", "bf16-mixed") != "bf16-mixed":
            raise ValueError("CPU EMA recipe requires bf16-mixed / FP32 parameters")
        return CPUPowerFunctionEMA(net, stds) if fabric.global_rank == 0 else None
    return PowerFunctionEMA(net, stds=stds,
        foreach=bool(config.get("performance", {}).get("ema_foreach", True)))


@contextmanager
def ema_evaluation(ema):
    if isinstance(ema, CPUPowerFunctionEMA):
        with ema.average_parameters() as model:
            yield model
    else:
        yield ema.get()[0][0]
