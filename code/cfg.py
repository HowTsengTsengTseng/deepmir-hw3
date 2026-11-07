"""Classifier-free guidance (CFG) for Stable Audio Open."""

def cfg(noise_pred_uncond, noise_pred_cond, guidance_scale):
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
