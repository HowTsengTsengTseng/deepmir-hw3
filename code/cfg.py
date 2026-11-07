"""Classifier-free guidance (CFG) and guidance-mode ablation variants for Stable Audio Open.

Supported guidance modes (`cfg_mode`) and schedule variants:
- `standard`:  u + w * (c - u)
- `uncond`:    u (equivalent to guidance_scale=0.0)
- `cond`:      c (equivalent to guidance_scale=1.0)
- `rescaled`:  CFG with standard-deviation rescaling (Lin et al., 2024; phi=rescale_phi)
- `apg`:       Adaptive Projected Guidance (Sadat et al., ICLR 2025): decomposes (c - u) into
               parallel and orthogonal components relative to c, scaling the parallel component
               by `eta` to suppress oversaturation at high guidance scales, with optional momentum.

Timestep scheduling (`cfg_schedule`):
- `constant`:  w(s) = w
- `linear`:    w(s) = 1 + (w - 1) * s          (ramp up as denoising progresses, s in [0, 1])
- `cosine`:    w(s) = 1 + (w - 1) * 0.5 * (1 - cos(pi * s))
- `interval`:  apply w when s in [interval_min, interval_max], else 1.0 (Kynkäänniemi et al., 2024)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

CFG_MODES = ("standard", "uncond", "cond", "rescaled", "apg")
CFG_SCHEDULES = ("constant", "linear", "cosine", "interval")


def cfg(noise_pred_uncond: torch.Tensor, noise_pred_cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Standard classifier-free guidance (matches diffusers StableAudioPipeline)."""
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


def cfg_rescaled(
    noise_pred_uncond: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    guidance_scale: float,
    rescale_phi: float = 0.7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rescaled CFG (Common Diffusion Noise Schedules and Sample Steps are Flawed, Lin et al. 2024)."""
    pred_cfg = cfg(noise_pred_uncond, noise_pred_cond, guidance_scale)
    if rescale_phi == 0.0:
        return pred_cfg
    dims = list(range(1, noise_pred_cond.ndim))
    std_cond = noise_pred_cond.std(dim=dims, keepdim=True)
    std_cfg = pred_cfg.std(dim=dims, keepdim=True).clamp(min=eps)
    pred_rescaled = pred_cfg * (std_cond / std_cfg)
    return rescale_phi * pred_rescaled + (1.0 - rescale_phi) * pred_cfg


class APGState:
    """Running momentum buffer for Adaptive Projected Guidance (Sadat et al., ICLR 2025)."""

    def __init__(self, momentum: float = -0.5):
        self.momentum = momentum
        self.running_avg: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.running_avg = None

    def update(self, diff: torch.Tensor) -> torch.Tensor:
        if self.momentum == 0.0:
            return diff
        if self.running_avg is None:
            self.running_avg = diff
        else:
            self.running_avg = diff + self.momentum * self.running_avg
        return self.running_avg


def cfg_apg(
    noise_pred_uncond: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    guidance_scale: float,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
    state: Optional[APGState] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Adaptive Projected Guidance (Eliminating Oversaturation and Artifacts of High Guidance Scales, ICLR 2025).

    Projects the update direction delta = c - u onto the conditional prediction c:
        delta = delta_parallel + delta_orth
        pred  = c + (w - 1) * (eta * delta_parallel + delta_orth)
    Note: when eta=1, norm_threshold=0, and momentum=0, this reduces identically to standard CFG.
    """
    diff = noise_pred_cond - noise_pred_uncond
    if state is not None:
        diff = state.update(diff)

    dims = list(range(1, diff.ndim))
    if norm_threshold > 0.0:
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True).clamp(min=eps)
        scale = torch.minimum(torch.ones_like(diff_norm), torch.full_like(diff_norm, norm_threshold) / diff_norm)
        diff = diff * scale

    cond_unit = noise_pred_cond / noise_pred_cond.norm(p=2, dim=dims, keepdim=True).clamp(min=eps)
    parallel = (diff * cond_unit).sum(dim=dims, keepdim=True) * cond_unit
    orthogonal = diff - parallel
    update = eta * parallel + orthogonal
    return noise_pred_cond + (guidance_scale - 1.0) * update


def scheduled_guidance_scale(
    base_scale: float,
    step_index: int,
    num_steps: int,
    schedule: str = "constant",
    interval_min: float = 0.2,
    interval_max: float = 0.8,
) -> float:
    """Compute effective guidance scale at `step_index` in [0, num_steps - 1]."""
    if schedule == "constant" or num_steps <= 1:
        return float(base_scale)
    s = step_index / float(num_steps - 1)  # 0.0 at start (noisy), 1.0 at end (clean)
    if schedule == "linear":
        return 1.0 + (base_scale - 1.0) * s
    if schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 - math.cos(math.pi * s))
    if schedule == "interval":
        return float(base_scale) if (interval_min <= s <= interval_max) else 1.0
    raise ValueError(f"Unknown cfg_schedule '{schedule}', expected one of {CFG_SCHEDULES}")


def apply_cfg(
    noise_pred_uncond: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    guidance_scale: float,
    *,
    mode: str = "standard",
    schedule: str = "constant",
    step_index: int = 0,
    num_steps: int = 1,
    rescale_phi: float = 0.7,
    apg_eta: float = 0.0,
    apg_norm_threshold: float = 0.0,
    apg_state: Optional[APGState] = None,
    interval_min: float = 0.2,
    interval_max: float = 0.8,
) -> torch.Tensor:
    """Unified CFG entry point supporting mode & timestep-schedule ablations."""
    w = scheduled_guidance_scale(
        guidance_scale,
        step_index=step_index,
        num_steps=num_steps,
        schedule=schedule,
        interval_min=interval_min,
        interval_max=interval_max,
    )
    if mode == "standard":
        return cfg(noise_pred_uncond, noise_pred_cond, w)
    if mode == "uncond":
        return noise_pred_uncond
    if mode == "cond":
        return noise_pred_cond
    if mode == "rescaled":
        return cfg_rescaled(noise_pred_uncond, noise_pred_cond, w, rescale_phi=rescale_phi)
    if mode == "apg":
        return cfg_apg(
            noise_pred_uncond,
            noise_pred_cond,
            w,
            eta=apg_eta,
            norm_threshold=apg_norm_threshold,
            state=apg_state,
        )
    raise ValueError(f"Unknown cfg_mode '{mode}', expected one of {CFG_MODES}")
