"""LAION-CLAP wrapper shared by retrieval (retrieve.py) and the CLAP score (eval.py).

The homework requires `music_audioset_epoch_15_esc_90.14.pt` (HTSAT-base audio tower, no fusion).
Audio preprocessing follows av-benchmark's CLAP extraction: mono, DC removal, resample to 48 kHz,
truncate to `audio_length` seconds, then laion_clap repeat-pads to 10 s (480000 samples).
"""

from __future__ import annotations

from functools import partial
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import WEIGHTS_DIR, FeatureCache, fix_length, iter_audio_batches, load_audio

log = logging.getLogger(__name__)

CLAP_CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_HF_REPO = "lukewys/laion_clap"
CLAP_SR = 48_000
CLAP_MAX_SAMPLES = 480_000  # 10 s; laion_clap's fixed input length without fusion


def resolve_clap_ckpt(path: Optional[Path] = None) -> Path:
    """Return a local checkpoint path, downloading the homework checkpoint from HF if needed."""
    path = Path(path) if path else WEIGHTS_DIR / CLAP_CKPT_NAME
    if path.exists():
        return path
    if path.name != CLAP_CKPT_NAME:
        raise FileNotFoundError(path)
    log.info(f"Downloading {CLAP_CKPT_NAME} from huggingface.co/{CLAP_HF_REPO} to {path.parent}")
    from huggingface_hub import hf_hub_download
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(CLAP_HF_REPO, CLAP_CKPT_NAME, local_dir=str(path.parent)))


def load_clap_waveform(path: Path, audio_length: float = 10.0) -> torch.Tensor:
    wav = load_audio(path, CLAP_SR, remove_dc=True, max_seconds=min(audio_length, 10.0))
    # identical to laion_clap's `repeatpad` for clips <= 10 s; done here so batches can be stacked
    return fix_length(wav, CLAP_MAX_SAMPLES, mode="repeat")


class ClapEncoder:
    """Thin wrapper around laion_clap.CLAP_Module returning L2-normalised CPU float embeddings."""

    def __init__(self, ckpt: Optional[Path] = None, device: str = "cuda", amodel: str = "HTSAT-base"):
        import laion_clap
        ckpt = resolve_clap_ckpt(ckpt)
        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=device)
        self.model.load_ckpt(str(ckpt), verbose=False)
        self.model.eval()
        self.ckpt_name = Path(ckpt).name
        log.info(f"Loaded LAION-CLAP {amodel} from {ckpt}")

    @torch.inference_mode()
    def embed_text(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            emb = self.model.get_text_embedding(chunk, use_tensor=True)
            out.append(emb.reshape(len(chunk), -1).float().cpu())
        return F.normalize(torch.cat(out), dim=-1)

    @torch.inference_mode()
    def embed_waveforms(self, wavs: torch.Tensor) -> torch.Tensor:
        """wavs: (B, 480000) at 48 kHz."""
        emb = self.model.get_audio_embedding_from_data(wavs.to(self.device).float(), use_tensor=True)
        return F.normalize(emb.reshape(wavs.shape[0], -1).float().cpu(), dim=-1)


def embed_audio_files(encoder, files: Dict[str, Path], *, cache: Optional[FeatureCache] = None,
                      audio_length: float = 10.0, batch_size: int = 32, num_workers: int = 0,
                      desc: str = "CLAP audio") -> Dict[str, torch.Tensor]:
    """{key: path} -> {key: (D,) normalised embedding}. Uses/updates `cache` when given.

    `encoder` only needs an `embed_waveforms((B, 480000)) -> (B, D)` method (mockable in tests).
    """
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
        load_fn = partial(load_clap_waveform, audio_length=audio_length)
        n_batches = -(-len(todo) // batch_size)
        for batch in tqdm(iter_audio_batches(todo, load_fn, batch_size, num_workers), total=n_batches, desc=desc):
            keys = [k for k, _ in batch]
            emb = encoder.embed_waveforms(torch.stack([w for _, w in batch]))
            for k, e in zip(keys, emb):
                out[k] = e.clone()
                if cache is not None:
                    cache.put(files[k], out[k])
        if cache is not None:
            cache.save()
    missing = set(files) - set(out)
    if missing:
        log.warning(f"{desc}: {len(missing)} files could not be embedded: {sorted(missing)[:10]} ...")
    return out


def stack_embeddings(emb: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.stack([emb[k] for k in keys])
