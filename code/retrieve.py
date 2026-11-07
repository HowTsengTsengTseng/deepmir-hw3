"""Text-to-audio retrieval: find the closest MusicCaps clip to each test prompt with LAION-CLAP.

    python code/retrieve.py                                   # defaults: test.csv vs data/audio_mp3
    python code/retrieve.py --topk 5 --device cuda --num_workers 8

Outputs (in --out_dir, default outputs/retrieval/):
    retrieval.json        {"<test id>": "<MusicCaps ytid>"}  (keys and values are strings; submission format)
    retrieval_topk.csv    id, rank, ytid, cosine similarity, caption  (for analysis / report)

The MusicCaps CLAP audio embeddings are cached in outputs/cache/ and reused by eval.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from clap_model import CLAP_CKPT_NAME, ClapEncoder, embed_audio_files, stack_embeddings
from common import (CACHE_DIR, MUSICCAPS_AUDIO_DIR, OUTPUT_DIR, TEST_CSV, FeatureCache, list_audio_files,
                    load_captions, pick_device, setup_logging)

log = logging.getLogger(__name__)


def clap_cache_path(ckpt_name: str = CLAP_CKPT_NAME) -> Path:
    return CACHE_DIR / f"clap_audio_{Path(ckpt_name).stem}.pt"


def rank_audio(text_emb: torch.Tensor, audio_emb: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine-similarity ranking. text_emb (Q, D), audio_emb (N, D) -> (scores, indices), each (Q, k)."""
    text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)
    audio_emb = torch.nn.functional.normalize(audio_emb.float(), dim=-1)
    sim = text_emb @ audio_emb.T
    k = min(topk, audio_emb.shape[0])
    return sim.topk(k, dim=-1)  # sorted descending; ties broken by index order


def build_results(query_ids: List[str], audio_keys: List[str], scores: torch.Tensor, indices: torch.Tensor,
                  with_ext: bool = False) -> Tuple[Dict[str, str], List[dict]]:
    """-> (submission mapping {id: ytid}, top-k rows)."""
    mapping, rows = {}, []
    for qi, qid in enumerate(query_ids):
        for rank, (s, ai) in enumerate(zip(scores[qi].tolist(), indices[qi].tolist()), start=1):
            key = audio_keys[ai]
            value = f"{key}.mp3" if with_ext else key
            if rank == 1:
                mapping[str(qid)] = str(value)
            rows.append({"id": qid, "rank": rank, "ytid": key, "score": round(float(s), 6)})
    return mapping, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captions", type=Path, default=TEST_CSV)
    p.add_argument("--audio_dir", type=Path, default=MUSICCAPS_AUDIO_DIR)
    p.add_argument("--out_dir", type=Path, default=OUTPUT_DIR / "retrieval")
    p.add_argument("--clap_ckpt", type=Path, default=None, help=f"default: weights/{CLAP_CKPT_NAME}")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--audio_length", type=float, default=10.0)
    p.add_argument("--with_ext", action="store_true", help="write 'ytid.mp3' instead of 'ytid' in retrieval.json")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    setup_logging()

    captions = load_captions(args.captions)
    audio_files = list_audio_files(args.audio_dir)
    log.info(f"{len(captions)} prompts, {len(audio_files)} MusicCaps clips in {args.audio_dir}")
    if not audio_files:
        raise SystemExit(f"No audio found in {args.audio_dir}")

    encoder = ClapEncoder(args.clap_ckpt, device=pick_device(args.device))
    cache = FeatureCache(clap_cache_path(encoder.ckpt_name))
    audio_emb = embed_audio_files(encoder, audio_files, cache=cache, audio_length=args.audio_length,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  desc="CLAP MusicCaps")
    audio_keys = [k for k in audio_files if k in audio_emb]
    ids = list(captions)
    text_emb = encoder.embed_text([captions[i] for i in ids])

    scores, indices = rank_audio(text_emb, stack_embeddings(audio_emb, audio_keys), args.topk)
    mapping, rows = build_results(ids, audio_keys, scores, indices, with_ext=args.with_ext)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "retrieval.json", "w") as f:
        json.dump(mapping, f, indent=2)
    with open(args.out_dir / "retrieval_topk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "rank", "ytid", "score", "caption"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "caption": captions[r["id"]] if r["rank"] == 1 else ""})

    top1 = scores[:, 0]
    n_unique = len(set(mapping.values()))
    log.info(f"Top-1 CLAP similarity: mean {top1.mean():.4f}, min {top1.min():.4f}, max {top1.max():.4f}; "
             f"{n_unique} unique clips for {len(mapping)} prompts")
    log.info(f"Wrote {args.out_dir / 'retrieval.json'} and retrieval_topk.csv")


if __name__ == "__main__":
    main()
