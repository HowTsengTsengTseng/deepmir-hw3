"""Evaluation pipeline for HW3 using `third_party/av-benchmark` metrics:
- FAD / FD-VGG:       Fréchet Distance on VGGish 128-D embeddings (0.96s patches at 16 kHz)
- FD-PaSST:           Fréchet Distance on PaSST 768-D embeddings (10s at 32 kHz)
- KL-PaSST-softmax:   Paired KL(GT || pred) over PaSST 527 AudioSet class logits
- CLAP-Score:         Mean cosine similarity between test prompts and generated audio using
                      LAION-CLAP (`music_audioset_epoch_15_esc_90.14.pt`, as required by HW3)

Why this script exists alongside `third_party/av-benchmark/evaluate.py`:
1. MusicCaps reference tracks in `data/audio_mp3/` are `.mp3` named by YouTube ID (`<ytid>.mp3`),
   while `av-benchmark` only globs `*.wav`/`*.flac` and expects identical stems (`0`..`99`) for
   paired KL. `eval.py` uses the retrieval mapping (`retrieval.json`) to pair each prompt `i` with
   its closest MusicCaps reference track for KL-PaSST, while using either all MusicCaps tracks or
   the retrieved subset as the background reference distribution for FAD.
2. Uses the HW3-mandated checkpoint `music_audioset_epoch_15_esc_90.14.pt` and evaluates only the
   required metrics (VGGish, PaSST, LAION-CLAP) without requiring ImageBind or Synchformer weights.
3. Supports evaluating multiple prediction directories in one command (e.g., CFG ablation sweeps)
   and outputs both JSON and a Markdown/CSV comparison table.

Examples:
    # Evaluate a single generation directory
    python code/eval.py --pred_dir outputs/gen/cfg_standard_s7

    # Evaluate a CFG ablation sweep and write a comparison table
    python code/eval.py --pred_dir outputs/gen/cfg_standard_s1 outputs/gen/cfg_standard_s3 \
                        outputs/gen/cfg_standard_s7 outputs/gen/cfg_apg_s7 \
                        --out outputs/eval/ablation_summary.json
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from clap_model import CLAP_CKPT_NAME, ClapEncoder, embed_audio_files, stack_embeddings
from common import (
    CACHE_DIR,
    MUSICCAPS_AUDIO_DIR,
    OUTPUT_DIR,
    TEST_CSV,
    FeatureCache,
    add_av_bench_to_path,
    fix_length,
    iter_audio_batches,
    list_audio_files,
    load_audio,
    load_captions,
    pick_device,
    setup_logging,
)
from retrieve import clap_cache_path

log = logging.getLogger(__name__)

VGGISH_SR = 16_000
PASST_SR = 32_000
PASST_SAMPLES = 320_000  # 10 s at 32 kHz (av-benchmark pads/truncates to 320000)


# ----------------------------------------------------------------------- av-benchmark metric imports
def compute_fd(pred_features: np.ndarray, gt_features: np.ndarray) -> float:
    add_av_bench_to_path()
    from scipy import linalg
    import av_bench.metrics.fad as _fad

    pred_arr = np.asarray(pred_features, dtype=np.float64)
    gt_arr = np.asarray(gt_features, dtype=np.float64)
    try:
        return float(_fad.compute_fd(pred_arr, gt_arr))
    except TypeError:
        # SciPy >= 1.15 removed `disp=False` from `scipy.linalg.sqrtm`; patch transparently
        orig_sqrtm = _fad.linalg.sqrtm

        def _compat_sqrtm(a, disp=False):
            return (orig_sqrtm(a), None) if disp is False else orig_sqrtm(a)

        _fad.linalg.sqrtm = _compat_sqrtm
        try:
            return float(_fad.compute_fd(pred_arr, gt_arr))
        finally:
            _fad.linalg.sqrtm = orig_sqrtm


def compute_kl_passt(pred_logits: torch.Tensor, gt_logits: torch.Tensor) -> Dict[str, float]:
    add_av_bench_to_path()
    from av_bench.metrics.kl import compute_kl as _av_compute_kl

    return _av_compute_kl([pred_logits.float()], gt_logits.float())


def compute_clap_score(text_emb: torch.Tensor, audio_emb: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """Return (mean_cosine_similarity, per_sample_cosine_similarities)."""
    per_sample = torch.cosine_similarity(text_emb.float(), audio_emb.float(), dim=-1)
    return float(per_sample.mean().item()), per_sample


# ----------------------------------------------------------------------------- feature extractors
def load_vggish_waveform(path: Path, audio_length: float = 10.0) -> torch.Tensor:
    return load_audio(path, VGGISH_SR, remove_dc=True, max_seconds=audio_length)


def load_passt_waveform(path: Path, audio_length: float = 10.0) -> torch.Tensor:
    wav = load_audio(path, PASST_SR, remove_dc=True, max_seconds=audio_length)
    return fix_length(wav, PASST_SAMPLES, mode="zero")


class VGGishExtractor:
    """Wraps `av_bench.vggish.vggish.VGGish(postprocess=False)` exactly as in av-benchmark."""

    def __init__(self, device: str = "cuda"):
        add_av_bench_to_path()
        from av_bench.vggish.vggish import VGGish

        self.device = device
        self.model = VGGish(device=device, pretrained=True, preprocess=True, postprocess=False).eval()

    @torch.inference_mode()
    def extract_one(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """wav_16k: (T,) -> (num_patches, 128) float32 CPU tensor."""
        out = self.model(wav_16k.unsqueeze(0))  # (1, num_patches, 128)
        return out.squeeze(0).float().cpu()


class PaSSTExtractor:
    """Wraps `hear21passt.base.get_basic_model(mode='all')` exactly as in av-benchmark."""

    def __init__(self, device: str = "cuda"):
        from hear21passt.base import get_basic_model

        self.device = device
        self.model = get_basic_model(mode="all").to(device).eval()

    @torch.inference_mode()
    def extract_batch(self, wavs_32k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """wavs_32k: (B, 320000) -> (logits (B, 527), embed (B, 768)) on CPU."""
        out = self.model(wavs_32k.to(self.device).float()).float().cpu()
        logits = out[:, :527]
        embed = out[:, 527:]
        return logits, embed


def extract_vggish_features(
    extractor,
    files: Dict[str, Path],
    *,
    cache: Optional[FeatureCache] = None,
    audio_length: float = 10.0,
    batch_size: int = 32,
    num_workers: int = 0,
    desc: str = "VGGish",
) -> Dict[str, torch.Tensor]:
    """Return {key: (num_patches, 128)} for each file in `files`."""
    out: Dict[str, torch.Tensor] = {}
    todo = []
    for key, path in files.items():
        hit = cache.get(path) if cache is not None else None
        if hit is not None:
            out[key] = hit
        else:
            todo.append((key, path))
    if todo:
        log.info(f"{desc}: extracting {len(todo)} / {len(files)} files ({len(files) - len(todo)} cached)")
        load_fn = partial(load_vggish_waveform, audio_length=audio_length)
        n_batches = -(-len(todo) // batch_size)
        for batch in tqdm(iter_audio_batches(todo, load_fn, batch_size, num_workers), total=n_batches, desc=desc):
            for key, wav in batch:
                try:
                    feat = extractor.extract_one(wav)
                except Exception as e:
                    log.error(f"VGGish failed on {key}: {e}")
                    continue
                out[key] = feat
                if cache is not None:
                    cache.put(files[key], feat)
        if cache is not None:
            cache.save()
    return out


def extract_passt_features(
    extractor,
    files: Dict[str, Path],
    *,
    cache: Optional[FeatureCache] = None,
    audio_length: float = 10.0,
    batch_size: int = 16,
    num_workers: int = 0,
    desc: str = "PaSST",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return ({key: logits (527,)}, {key: embed (768,)}) for each file in `files`."""
    logits_dict: Dict[str, torch.Tensor] = {}
    embed_dict: Dict[str, torch.Tensor] = {}
    todo = []
    for key, path in files.items():
        hit = cache.get(path) if cache is not None else None
        if hit is not None:
            logits_dict[key], embed_dict[key] = hit["logits"], hit["embed"]
        else:
            todo.append((key, path))
    if todo:
        log.info(f"{desc}: extracting {len(todo)} / {len(files)} files ({len(files) - len(todo)} cached)")
        load_fn = partial(load_passt_waveform, audio_length=audio_length)
        n_batches = -(-len(todo) // batch_size)
        for batch in tqdm(iter_audio_batches(todo, load_fn, batch_size, num_workers), total=n_batches, desc=desc):
            keys = [k for k, _ in batch]
            wavs = torch.stack([w for _, w in batch])
            logits, embeds = extractor.extract_batch(wavs)
            for k, l_vec, e_vec in zip(keys, logits, embeds):
                l_c, e_c = l_vec.clone(), e_vec.clone()
                logits_dict[k] = l_c
                embed_dict[k] = e_c
                if cache is not None:
                    cache.put(files[k], {"logits": l_c, "embed": e_c})
        if cache is not None:
            cache.save()
    return logits_dict, embed_dict


# ------------------------------------------------------------------- pairing & metric orchestration
def pair_gt_and_pred_keys(
    pred_keys: Sequence[str],
    gt_available_keys: Sequence[str],
    retrieval_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Return list of `(pred_key, gt_key)` pairs for paired metrics (KL-PaSST).

    1. If `pred_key` matches a key in `gt_available_keys` directly (e.g., GT directory already uses
       stems `0`..`99`), pair `(pred_key, pred_key)`.
    2. Otherwise, if `retrieval_map` is provided (maps `id -> ytid` or `id -> ytid.mp3`), pair
       `(pred_key, Path(retrieval_map[pred_key]).stem)` when present in `gt_available_keys`.
    """
    gt_set = set(gt_available_keys)
    pairs: List[Tuple[str, str]] = []
    for pk in pred_keys:
        if pk in gt_set:
            pairs.append((pk, pk))
        elif retrieval_map and pk in retrieval_map:
            mapped_stem = Path(str(retrieval_map[pk])).stem
            if mapped_stem in gt_set:
                pairs.append((pk, mapped_stem))
    return pairs


def evaluate_features(
    *,
    pred_vggish: Optional[Dict[str, torch.Tensor]] = None,
    gt_vggish: Optional[Dict[str, torch.Tensor]] = None,
    pred_passt_logits: Optional[Dict[str, torch.Tensor]] = None,
    gt_passt_logits: Optional[Dict[str, torch.Tensor]] = None,
    pred_passt_embed: Optional[Dict[str, torch.Tensor]] = None,
    gt_passt_embed: Optional[Dict[str, torch.Tensor]] = None,
    pred_clap_audio: Optional[Dict[str, torch.Tensor]] = None,
    text_clap_emb: Optional[Dict[str, torch.Tensor]] = None,
    retrieval_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Pure metric computation over extracted feature dicts ( matches av-benchmark math )."""
    metrics: Dict[str, float] = {}

    if pred_vggish and gt_vggish:
        # av_bench/evaluate.py L121-122: unroll_dict(..., cat=True) pools all 0.96s patches
        pred_mat = torch.cat(list(pred_vggish.values()), dim=0).numpy()
        gt_mat = torch.cat(list(gt_vggish.values()), dim=0).numpy()
        metrics["FD-VGG"] = compute_fd(pred_mat, gt_mat)

    if pred_passt_embed and gt_passt_embed:
        pred_mat = torch.stack(list(pred_passt_embed.values()), dim=0).numpy()
        gt_mat = torch.stack(list(gt_passt_embed.values()), dim=0).numpy()
        metrics["FD-PASST"] = compute_fd(pred_mat, gt_mat)

    if pred_passt_logits and gt_passt_logits:
        pairs = pair_gt_and_pred_keys(list(pred_passt_logits.keys()), list(gt_passt_logits.keys()), retrieval_map)
        if pairs:
            p_logits = torch.stack([pred_passt_logits[p] for p, _ in pairs], dim=0)
            g_logits = torch.stack([gt_passt_logits[g] for _, g in pairs], dim=0)
            kl = compute_kl_passt(p_logits, g_logits)
            metrics["KL-PASST-softmax"] = kl["kl_softmax"]
            metrics["KL-PASST-sigmoid"] = kl["kl_sigmoid"]
            metrics["KL-num-pairs"] = float(len(pairs))
        else:
            log.warning("KL-PaSST skipped: no paired keys between predictions and GT (pass --retrieval_json)")

    if pred_clap_audio and text_clap_emb:
        common = [k for k in pred_clap_audio if k in text_clap_emb]
        if common:
            t_mat = stack_embeddings(text_clap_emb, common)
            a_mat = stack_embeddings(pred_clap_audio, common)
            mean_score, _ = compute_clap_score(t_mat, a_mat)
            metrics["CLAP-Score"] = mean_score
        else:
            log.warning("CLAP-Score skipped: no matching IDs between predictions and captions CSV")

    return metrics


def format_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format a Markdown table comparing multiple evaluated runs (e.g., CFG ablation)."""
    cols = ["FD-VGG", "FD-PASST", "KL-PASST-softmax", "CLAP-Score"]
    present_cols = [c for c in cols if any(c in m for m in results.values())]
    header = "| Run | " + " | ".join(present_cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(present_cols)) + "|"
    lines = [header, sep]
    for name, m in results.items():
        vals = [f"{m[c]:.4f}" if c in m else "N/A" for c in present_cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pred_dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories of generated audio (<id>.wav/.mp3/.flac) to evaluate",
    )
    p.add_argument("--gt_dir", type=Path, default=MUSICCAPS_AUDIO_DIR, help="Reference audio directory (MusicCaps)")
    p.add_argument("--captions", type=Path, default=TEST_CSV, help="Test captions CSV (id,caption)")
    p.add_argument(
        "--retrieval_json",
        type=Path,
        default=OUTPUT_DIR / "retrieval" / "retrieval.json",
        help="Retrieval mapping {id: ytid} used to pair each prompt with its closest MusicCaps clip for KL",
    )
    p.add_argument(
        "--gt_subset",
        choices=["retrieved", "all"],
        default="retrieved",
        help="Use only the 100 retrieved MusicCaps clips ('retrieved') or all 5367 clips ('all') for FAD & GT extraction",
    )
    p.add_argument("--audio_length", type=float, default=10.0, help="Truncate audio at this length in seconds")
    p.add_argument("--clap_ckpt", type=Path, default=None, help=f"Default: weights/{CLAP_CKPT_NAME}")
    p.add_argument("--cache_dir", type=Path, default=CACHE_DIR)
    p.add_argument("--out", type=Path, default=None, help="Optional summary JSON output path")
    p.add_argument("--skip_fad", action="store_true")
    p.add_argument("--skip_kl", action="store_true")
    p.add_argument("--skip_clap", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    args = build_parser().parse_args(argv)
    setup_logging()
    device = pick_device(args.device)

    captions = load_captions(args.captions) if args.captions and args.captions.exists() else {}
    all_gt_files = list_audio_files(args.gt_dir) if (not args.skip_fad or not args.skip_kl) else {}

    retrieval_map: Optional[Dict[str, str]] = None
    if args.retrieval_json and args.retrieval_json.exists():
        with open(args.retrieval_json) as f:
            retrieval_map = json.load(f)
        log.info(f"Loaded retrieval mapping ({len(retrieval_map)} pairs) from {args.retrieval_json}")
    elif not args.skip_kl:
        # Auto-compute retrieval mapping if CLAP is available and GT stems don't directly match 0..99
        sample_pred_files = list_audio_files(args.pred_dir[0])
        direct_overlap = set(sample_pred_files) & set(all_gt_files)
        if not direct_overlap and captions and all_gt_files:
            log.info("retrieval.json not found; computing CLAP retrieval mapping on the fly...")
            from retrieve import build_results, rank_audio

            clap_enc = ClapEncoder(args.clap_ckpt, device=device)
            clap_cache = FeatureCache(args.cache_dir / f"clap_audio_{Path(clap_enc.ckpt_name).stem}.pt")
            mc_emb = embed_audio_files(
                clap_enc,
                all_gt_files,
                cache=clap_cache,
                audio_length=args.audio_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc="CLAP MusicCaps",
            )
            mc_keys = [k for k in all_gt_files if k in mc_emb]
            q_ids = list(captions)
            t_emb = clap_enc.embed_text([captions[i] for i in q_ids])
            scores, indices = rank_audio(t_emb, stack_embeddings(mc_emb, mc_keys), topk=1)
            retrieval_map, _ = build_results(q_ids, mc_keys, scores, indices)
            if args.retrieval_json:
                args.retrieval_json.parent.mkdir(parents=True, exist_ok=True)
                with open(args.retrieval_json, "w") as f:
                    json.dump(retrieval_map, f, indent=2)
                log.info(f"Saved retrieval mapping to {args.retrieval_json}")

    # Select which GT files need VGGish / PaSST extraction
    if args.gt_subset == "retrieved" and retrieval_map:
        needed_stems = {Path(str(v)).stem for v in retrieval_map.values()}
        gt_files = {k: p for k, p in all_gt_files.items() if k in needed_stems}
        log.info(f"Using {len(gt_files)} retrieved MusicCaps reference tracks for GT features (--gt_subset retrieved)")
    else:
        gt_files = all_gt_files
        if gt_files:
            log.info(f"Using all {len(gt_files)} reference tracks in {args.gt_dir} for GT features")

    vggish_cache = FeatureCache(args.cache_dir / f"vggish_len{args.audio_length:g}.pt")
    passt_cache = FeatureCache(args.cache_dir / f"passt_len{args.audio_length:g}.pt")

    vggish_ext = VGGishExtractor(device=device) if not args.skip_fad else None
    need_passt = (not args.skip_fad) or (not args.skip_kl)
    passt_ext = PaSSTExtractor(device=device) if need_passt else None

    gt_vggish = (
        extract_vggish_features(
            vggish_ext,
            gt_files,
            cache=vggish_cache,
            audio_length=args.audio_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc="VGGish GT",
        )
        if vggish_ext and gt_files
        else None
    )
    gt_passt_logits, gt_passt_embed = (
        extract_passt_features(
            passt_ext,
            gt_files,
            cache=passt_cache,
            audio_length=args.audio_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc="PaSST GT",
        )
        if passt_ext and gt_files
        else (None, None)
    )

    clap_enc = None
    text_clap_emb: Optional[Dict[str, torch.Tensor]] = None
    clap_cache = None
    if not args.skip_clap:
        clap_enc = ClapEncoder(args.clap_ckpt, device=device)
        clap_cache = FeatureCache(clap_cache_path(clap_enc.ckpt_name))
        q_ids = list(captions)
        t_vecs = clap_enc.embed_text([captions[i] for i in q_ids], batch_size=args.batch_size)
        text_clap_emb = {qid: vec for qid, vec in zip(q_ids, t_vecs)}

    all_results: Dict[str, Dict[str, float]] = {}
    for pred_dir in args.pred_dir:
        pred_dir = Path(pred_dir)
        run_name = pred_dir.name
        pred_files = list_audio_files(pred_dir)
        if not pred_files:
            log.warning(f"No audio files found in {pred_dir}, skipping")
            continue
        log.info(f"Evaluating '{run_name}' ({len(pred_files)} files in {pred_dir})")

        pred_vggish = (
            extract_vggish_features(
                vggish_ext,
                pred_files,
                cache=vggish_cache,
                audio_length=args.audio_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc=f"VGGish ({run_name})",
            )
            if vggish_ext
            else None
        )
        pred_passt_logits, pred_passt_embed = (
            extract_passt_features(
                passt_ext,
                pred_files,
                cache=passt_cache,
                audio_length=args.audio_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc=f"PaSST ({run_name})",
            )
            if passt_ext
            else (None, None)
        )
        pred_clap_audio = (
            embed_audio_files(
                clap_enc,
                pred_files,
                cache=clap_cache,
                audio_length=args.audio_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc=f"CLAP ({run_name})",
            )
            if clap_enc
            else None
        )

        metrics = evaluate_features(
            pred_vggish=pred_vggish if not args.skip_fad else None,
            gt_vggish=gt_vggish if not args.skip_fad else None,
            pred_passt_logits=pred_passt_logits if not args.skip_kl else None,
            gt_passt_logits=gt_passt_logits if not args.skip_kl else None,
            pred_passt_embed=pred_passt_embed if not args.skip_fad else None,
            gt_passt_embed=gt_passt_embed if not args.skip_fad else None,
            pred_clap_audio=pred_clap_audio if not args.skip_clap else None,
            text_clap_emb=text_clap_emb if not args.skip_clap else None,
            retrieval_map=retrieval_map,
        )

        # Attach generation config metadata if present
        cfg_file = pred_dir / "config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                gen_cfg = json.load(f)
            if "guidance_scale" in gen_cfg:
                metrics["guidance_scale"] = float(gen_cfg["guidance_scale"])

        for k, v in metrics.items():
            log.info(f"  {run_name} | {k:<18}: {v:.6f}")

        with open(pred_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        all_results[run_name] = metrics

    if len(all_results) > 1:
        table = format_comparison_table(all_results)
        print("\n" + table + "\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        csv_out = args.out.with_suffix(".csv")
        cols = sorted({k for m in all_results.values() for k in m})
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", *cols])
            w.writeheader()
            for run, m in all_results.items():
                w.writerow({"run": run, **m})
        log.info(f"Wrote summary to {args.out} and {csv_out}")

    return all_results


if __name__ == "__main__":
    main()
