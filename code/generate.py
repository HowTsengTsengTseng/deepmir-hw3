"""Stable Audio Open sampling loop with configurable CFG and batch/ablation generation support.

Single-prompt usage (backwards-compatible with sample code):
    python code/generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav
    python code/generate.py --prompt "jazz, electric guitar, piano, relaxing" --check

Dataset & CFG ablation usage (loads pipeline once, sweeps scales/modes/schedules across test.csv):
    python code/generate.py --csv test.csv --out_dir outputs/gen/cfg_7.0 --scale 7.0
    python code/generate.py --csv test.csv --out_dir outputs/gen --scales 1.0 3.0 5.0 7.0 10.0
    python code/generate.py --csv test.csv --out_dir outputs/gen/apg_7.0 --cfg_mode apg --scale 7.0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from cfg import APGState, CFG_MODES, CFG_SCHEDULES, apply_cfg, cfg

load_dotenv()


@torch.no_grad()
def generate(
    pipe,
    prompt: str,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 100,
    audio_end_in_s: float = 10.0,
    seed: int = 0,
    *,
    negative_prompt: Optional[str] = None,
    cfg_mode: str = "standard",
    cfg_schedule: str = "constant",
    rescale_phi: float = 0.7,
    apg_eta: float = 0.0,
    apg_momentum: float = -0.5,
    apg_norm_threshold: float = 0.0,
    interval_min: float = 0.2,
    interval_max: float = 0.8,
    step_records_out: Optional[list[dict[str, float]]] = None,
    decode_audio: bool = True,
) -> Optional[torch.Tensor]:
    device = pipe._execution_device
    sampling_rate = pipe.vae.config.sampling_rate

    # 1. Conditions: [uncond, cond]. By default the unconditional branch is an all-zero text+duration
    # embedding (matching sample code). When `negative_prompt` is given, encode it explicitly.
    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, audio_end_in_s, device, False, batch_size=1)
    cond = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
    if negative_prompt is None:
        uncond = torch.zeros_like(cond)
    else:
        neg_embeds = pipe.encode_prompt(negative_prompt, device, do_classifier_free_guidance=False)
        uncond = torch.cat([neg_embeds, seconds_start, seconds_end], dim=1)
    encoder_hidden_states = torch.cat([uncond, cond])
    global_hidden_states = torch.cat([seconds_start, seconds_end], dim=2).repeat(2, 1, 1)

    # 2. Timesteps, initial noise, rotary embedding
    generator = torch.Generator(device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        int(pipe.transformer.config.sample_size),
        cond.dtype,
        device,
        generator,
        audio_channels=pipe.vae.config.audio_channels,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim,
        latents.shape[2] + global_hidden_states.shape[1],
        use_real=True,
        repeat_interleave_real=False,
    )

    apg_state = APGState(momentum=apg_momentum) if cfg_mode == "apg" else None
    timesteps = pipe.scheduler.timesteps
    num_steps = len(timesteps)

    # 3. Denoising loop
    for step_idx, t in enumerate(pipe.progress_bar(timesteps)):
        latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = pipe.transformer(
            latent_model_input,
            t.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        if cfg_mode == "standard" and cfg_schedule == "constant":
            noise_pred = cfg(noise_pred_uncond, noise_pred_cond, guidance_scale)
        else:
            noise_pred = apply_cfg(
                noise_pred_uncond,
                noise_pred_cond,
                guidance_scale,
                mode=cfg_mode,
                schedule=cfg_schedule,
                step_index=step_idx,
                num_steps=num_steps,
                rescale_phi=rescale_phi,
                apg_eta=apg_eta,
                apg_norm_threshold=apg_norm_threshold,
                apg_state=apg_state,
                interval_min=interval_min,
                interval_max=interval_max,
            )
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    if step_records_out is not None and apg_state is not None:
        step_records_out.extend(apg_state.step_records)

    if not decode_audio:
        return None

    # 4. Decode
    audio = pipe.vae.decode(latents).sample
    return audio[0, :, : int(audio_end_in_s * sampling_rate)]


def save_audio(path: Path, audio: torch.Tensor, sampling_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # audio is (channels, samples)
    arr = audio.T.float().cpu().numpy()
    try:
        import soundfile as sf

        sf.write(str(path), arr, sampling_rate)
        return
    except Exception:
        pass
    try:
        import torchaudio

        torchaudio.save(str(path), audio.float().cpu(), sampling_rate)
        return
    except Exception:
        pass
    if path.suffix.lower() == ".mp3":
        import shutil
        import subprocess
        import tempfile
        from scipy.io import wavfile

        if shutil.which("ffmpeg"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = Path(tmp.name)
            try:
                pcm16 = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
                wavfile.write(str(tmp_wav), sampling_rate, pcm16)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_wav), "-b:a", "192k", str(path)],
                    check=True,
                )
                return
            finally:
                tmp_wav.unlink(missing_ok=True)
    from scipy.io import wavfile

    wavfile.write(str(path), sampling_rate, arr)


def format_ablation_name(
    scale: float,
    cfg_mode: str = "standard",
    cfg_schedule: str = "constant",
) -> str:
    tag = f"cfg_{cfg_mode}"
    if cfg_schedule != "constant":
        tag += f"_{cfg_schedule}"
    tag += f"_s{scale:g}"
    return tag


def summarize_apg_norms(all_prompt_records: dict[str, list[dict[str, float]]]) -> dict:
    """Aggregate per-step APG vector norms across prompts and denoising steps."""
    if not all_prompt_records:
        return {}
    keys = ("raw_diff_norm", "unclipped_norm", "clipped_norm", "parallel_norm", "orthogonal_norm", "cond_norm")
    flat: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [rec[k] for records in all_prompt_records.values() for rec in records if k in rec]
        flat[k] = np.asarray(vals, dtype=np.float64)

    u = flat["unclipped_norm"]
    if u.size == 0:
        return {}

    def _quantiles(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    num_steps = max(len(r) for r in all_prompt_records.values())
    per_step_unclipped = np.array(
        [[r[s]["unclipped_norm"] for r in all_prompt_records.values() if len(r) > s] for s in range(num_steps)],
        dtype=np.float64,
    )
    per_step_raw = np.array(
        [[r[s]["raw_diff_norm"] for r in all_prompt_records.values() if len(r) > s] for s in range(num_steps)],
        dtype=np.float64,
    )
    per_step_par = np.array(
        [[r[s]["parallel_norm"] for r in all_prompt_records.values() if len(r) > s] for s in range(num_steps)],
        dtype=np.float64,
    )
    per_step_orth = np.array(
        [[r[s]["orthogonal_norm"] for r in all_prompt_records.values() if len(r) > s] for s in range(num_steps)],
        dtype=np.float64,
    )
    per_step_cond = np.array(
        [[r[s]["cond_norm"] for r in all_prompt_records.values() if len(r) > s] for s in range(num_steps)],
        dtype=np.float64,
    )

    return {
        "num_prompts_logged": len(all_prompt_records),
        "num_steps_per_prompt": num_steps,
        "total_step_evaluations": int(u.size),
        "unclipped_norm_summary": _quantiles(u),
        "raw_diff_norm_summary": _quantiles(flat["raw_diff_norm"]),
        "clipped_norm_summary": _quantiles(flat["clipped_norm"]),
        "parallel_norm_summary": _quantiles(flat["parallel_norm"]),
        "orthogonal_norm_summary": _quantiles(flat["orthogonal_norm"]),
        "cond_norm_summary": _quantiles(flat["cond_norm"]),
        "clip_rate_at_thresholds": {
            "r=1.0": float(np.mean(u > 1.0)),
            "r=2.5": float(np.mean(u > 2.5)),
            "r=5.0": float(np.mean(u > 5.0)),
            "r=7.5": float(np.mean(u > 7.5)),
            "r=10.0": float(np.mean(u > 10.0)),
            "r=15.0": float(np.mean(u > 15.0)),
            "r=20.0": float(np.mean(u > 20.0)),
        },
        "per_step_trajectory": {
            "unclipped_norm_mean": [round(float(x), 4) for x in per_step_unclipped.mean(axis=1)],
            "unclipped_norm_max": [round(float(x), 4) for x in per_step_unclipped.max(axis=1)],
            "unclipped_norm_min": [round(float(x), 4) for x in per_step_unclipped.min(axis=1)],
            "raw_diff_norm_mean": [round(float(x), 4) for x in per_step_raw.mean(axis=1)],
            "parallel_norm_mean": [round(float(x), 4) for x in per_step_par.mean(axis=1)],
            "orthogonal_norm_mean": [round(float(x), 4) for x in per_step_orth.mean(axis=1)],
            "cond_norm_mean": [round(float(x), 4) for x in per_step_cond.mean(axis=1)],
        },
    }


def generate_dataset(
    pipe,
    captions: dict[str, str],
    out_dir: Path,
    *,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 100,
    audio_end_in_s: float = 10.0,
    seed: int = 0,
    overwrite: bool = False,
    negative_prompt: Optional[str] = None,
    cfg_mode: str = "standard",
    cfg_schedule: str = "constant",
    rescale_phi: float = 0.7,
    apg_eta: float = 0.0,
    apg_momentum: float = -0.5,
    apg_norm_threshold: float = 0.0,
    interval_min: float = 0.2,
    interval_max: float = 0.8,
    log_norms: bool = False,
) -> Path:
    """Generate one audio file `<id>.mp3` per row in `captions` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling_rate = pipe.vae.config.sampling_rate

    meta = {
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "audio_end_in_s": audio_end_in_s,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "cfg_mode": cfg_mode,
        "cfg_schedule": cfg_schedule,
        "rescale_phi": rescale_phi,
        "apg_eta": apg_eta,
        "apg_momentum": apg_momentum,
        "apg_norm_threshold": apg_norm_threshold,
        "interval_min": interval_min,
        "interval_max": interval_max,
        "num_prompts": len(captions),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    all_prompt_norms: dict[str, list[dict[str, float]]] = {}
    for qid, prompt in tqdm(captions.items(), desc=f"gen ({out_dir.name})"):
        out_path = out_dir / f"{qid}.mp3"
        need_audio = (not out_path.exists()) or overwrite
        if not need_audio and not log_norms:
            continue
        prompt_records: list[dict[str, float]] = []
        audio = generate(
            pipe,
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            audio_end_in_s=audio_end_in_s,
            seed=seed,
            negative_prompt=negative_prompt,
            cfg_mode=cfg_mode,
            cfg_schedule=cfg_schedule,
            rescale_phi=rescale_phi,
            apg_eta=apg_eta,
            apg_momentum=apg_momentum,
            apg_norm_threshold=apg_norm_threshold,
            interval_min=interval_min,
            interval_max=interval_max,
            step_records_out=prompt_records if cfg_mode == "apg" else None,
            decode_audio=need_audio,
        )
        if prompt_records:
            all_prompt_norms[qid] = prompt_records
        if need_audio and audio is not None:
            save_audio(out_path, audio, sampling_rate)

    if all_prompt_norms:
        norm_stats = summarize_apg_norms(all_prompt_norms)
        norm_path = out_dir / "norm_stats.json"
        with open(norm_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        u_sum = norm_stats["unclipped_norm_summary"]
        rates = norm_stats["clip_rate_at_thresholds"]
        print(
            f"[APG Norm Stats -> {norm_path}] "
            f"unclipped ||d_t||_2: min={u_sum['min']:.3f}, p25={u_sum['p25']:.3f}, "
            f"median={u_sum['median']:.3f}, mean={u_sum['mean']:.3f}±{u_sum['std']:.3f}, "
            f"p75={u_sum['p75']:.3f}, p95={u_sum['p95']:.3f}, max={u_sum['max']:.3f} | "
            f"clip_rate(r=2.5)={rates['r=2.5']*100:.1f}%, clip_rate(r=5.0)={rates['r=5.0']*100:.1f}%"
        )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default=None, help="Single prompt to generate (if --csv is not used)")
    parser.add_argument("--csv", type=Path, default=None, help="CSV with id,caption columns for batch generation")
    parser.add_argument("--scale", type=float, default=7.0, help="CFG guidance scale")
    parser.add_argument("--scales", type=float, nargs="+", default=None, help="Sweep multiple CFG scales (ablation)")
    parser.add_argument("--cfg_mode", choices=CFG_MODES, default="standard", help="CFG variant for ablation study")
    parser.add_argument("--cfg_schedule", choices=CFG_SCHEDULES, default="constant", help="Timestep schedule for CFG")
    parser.add_argument("--rescale_phi", type=float, default=0.7, help="Rescale factor phi for --cfg_mode rescaled")
    parser.add_argument("--apg_eta", type=float, default=0.0, help="Parallel component weight eta for --cfg_mode apg")
    parser.add_argument("--apg_momentum", type=float, default=-0.5, help="Momentum for --cfg_mode apg")
    parser.add_argument("--apg_norm_threshold", type=float, default=0.0, help="Gradient clipping norm for APG")
    parser.add_argument("--interval_min", type=float, default=0.2, help="Start progress [0,1] for interval schedule")
    parser.add_argument("--interval_max", type=float, default=0.8, help="End progress [0,1] for interval schedule")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Optional negative text prompt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out.mp3", help="Output file for single-prompt mode")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/gen"), help="Output directory for --csv mode")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log_norms",
        action="store_true",
        help="Track per-step APG vector norms into norm_stats.json even if audio files already exist in --out_dir",
    )
    parser.add_argument("--limit", type=int, default=None, help="Generate only first N rows of --csv")
    parser.add_argument("--check", action="store_true", help="compare with StableAudioPipeline")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompt is None and args.csv is None:
        parser.error("Must specify either --prompt or --csv")

    hf_token = os.getenv("HF_TOKEN")
    from diffusers import StableAudioPipeline

    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16, token=hf_token
    )
    pipe = pipe.to("cuda")

    if args.csv is not None:
        from common import load_captions

        captions = load_captions(args.csv)
        if args.limit is not None:
            captions = dict(list(captions.items())[: args.limit])

        scales = args.scales if args.scales is not None else [args.scale]
        for scale in scales:
            if args.scales is not None:
                sub_dir = args.out_dir / format_ablation_name(scale, args.cfg_mode, args.cfg_schedule)
            else:
                sub_dir = args.out_dir
            generate_dataset(
                pipe,
                captions,
                sub_dir,
                guidance_scale=scale,
                num_inference_steps=args.steps,
                audio_end_in_s=args.seconds,
                seed=args.seed,
                overwrite=args.overwrite,
                negative_prompt=args.negative_prompt,
                cfg_mode=args.cfg_mode,
                cfg_schedule=args.cfg_schedule,
                rescale_phi=args.rescale_phi,
                apg_eta=args.apg_eta,
                apg_momentum=args.apg_momentum,
                apg_norm_threshold=args.apg_norm_threshold,
                interval_min=args.interval_min,
                interval_max=args.interval_max,
                log_norms=args.log_norms,
            )
        return

    audio = generate(
        pipe,
        args.prompt,
        args.scale,
        args.steps,
        args.seconds,
        args.seed,
        negative_prompt=args.negative_prompt,
        cfg_mode=args.cfg_mode,
        cfg_schedule=args.cfg_schedule,
        rescale_phi=args.rescale_phi,
        apg_eta=args.apg_eta,
        apg_momentum=args.apg_momentum,
        apg_norm_threshold=args.apg_norm_threshold,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )
    save_audio(Path(args.out), audio, pipe.vae.config.sampling_rate)
    print(f"Saved to {args.out}")

    if args.check:
        ref = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            audio_end_in_s=args.seconds,
            guidance_scale=args.scale,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).audios[0]
        print(f"max |diffusers - ours| = {(ref.float() - audio.float()).abs().max().item():.3e} (0 = identical)")


if __name__ == "__main__":
    main()
