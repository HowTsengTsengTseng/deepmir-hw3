"""Shared helpers for retrieval / generation / evaluation: paths, captions, audio I/O, feature cache."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MUSICCAPS_AUDIO_DIR = DATA_DIR / "audio_mp3"
TEST_CSV = REPO_ROOT / "test.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
WEIGHTS_DIR = REPO_ROOT / "weights"
AV_BENCH_DIR = REPO_ROOT / "third_party" / "av-benchmark"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def add_av_bench_to_path() -> None:
    """Make `import av_bench` work without `pip install -e third_party/av-benchmark`."""
    try:
        import av_bench  # noqa: F401
    except ImportError:
        if not AV_BENCH_DIR.exists():
            raise FileNotFoundError(
                f"av-benchmark not found at {AV_BENCH_DIR}; "
                "git clone https://github.com/hkchengrex/av-benchmark.git third_party/av-benchmark")
        sys.path.insert(0, str(AV_BENCH_DIR))


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------------------------- data
def load_captions(csv_path: Path = TEST_CSV) -> Dict[str, str]:
    """Read `id,caption` (or `name,caption`) CSV -> ordered {id (str): caption}."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key_col = "id" if "id" in reader.fieldnames else "name"
        if key_col not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have columns id/name and caption, got {reader.fieldnames}")
        out = {str(row[key_col]).strip(): row["caption"].strip() for row in reader}
    return dict(sorted(out.items(), key=lambda kv: _natural_key(kv[0])))


def _natural_key(s: str):
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def list_audio_files(directory: Path) -> Dict[str, Path]:
    """{stem: path} for all `.mp3` files directly inside `directory` (naturally sorted by stem)."""
    directory = Path(directory)
    files = {p.stem: p for p in directory.glob("*.mp3") if p.is_file() and not p.name.startswith(".")}
    return dict(sorted(files.items(), key=lambda kv: _natural_key(kv[0])))


# ---------------------------------------------------------------------------------------- audio I/O
def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Return (channels, samples) float32 array and its sample rate."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
        return data.T, sr
    except Exception as e:  # older libsndfile without mp3 support, etc.
        try:
            import librosa
            data, sr = librosa.load(str(path), sr=None, mono=False)
            return np.atleast_2d(data).astype(np.float32), sr
        except ImportError:
            pass
        if Path(path).suffix.lower() in (".wav", ".mp3"):
            try:
                from scipy.io import wavfile
                sr, raw = wavfile.read(str(path))
                if raw.dtype == np.int16:
                    arr = raw.astype(np.float32) / 32768.0
                elif raw.dtype == np.int32:
                    arr = raw.astype(np.float32) / 2147483648.0
                elif raw.dtype == np.uint8:
                    arr = (raw.astype(np.float32) - 128.0) / 128.0
                else:
                    arr = raw.astype(np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                else:
                    arr = arr.T
                return arr, int(sr)
            except Exception:
                pass
        raise RuntimeError(f"Could not decode {path} with soundfile ({e}) and librosa is missing")


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio.functional as AF
        # same default kernel as torchaudio.transforms.Resample used by av-benchmark
        return AF.resample(wav, orig_sr, target_sr)
    except ImportError:
        import math
        from scipy.signal import resample_poly
        g = math.gcd(int(orig_sr), int(target_sr))
        arr = resample_poly(wav.detach().cpu().numpy(), int(target_sr) // g, int(orig_sr) // g, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(wav.device)


def load_audio(path: Path, sr: int, *, remove_dc: bool = True,
               max_seconds: Optional[float] = None) -> torch.Tensor:
    """Load as mono float32 at `sr`, optionally truncated to `max_seconds`.

    Mirrors av_bench.data.audio_dataset.AudioDataset.read_from_file:
    channel mean -> DC removal -> resample -> truncate (no padding).
    """
    data, file_sr = _read_audio(Path(path))
    wav = torch.from_numpy(np.ascontiguousarray(data)).mean(dim=0)
    if remove_dc:
        wav = wav - wav.mean()
    wav = resample(wav, file_sr, sr)
    if max_seconds is not None:
        wav = wav[:int(sr * max_seconds)]
    return wav.contiguous()


def load_audio_multi(path: Path, srs: Sequence[int], *, remove_dc: bool = True,
                     max_seconds: Optional[float] = None) -> List[torch.Tensor]:
    """Decode once, return one mono waveform per target sample rate (same processing as `load_audio`)."""
    data, file_sr = _read_audio(Path(path))
    wav = torch.from_numpy(np.ascontiguousarray(data)).mean(dim=0)
    if remove_dc:
        wav = wav - wav.mean()
    out = []
    for sr in srs:
        w = resample(wav, file_sr, sr)
        if max_seconds is not None:
            w = w[:int(sr * max_seconds)]
        out.append(w.contiguous())
    return out


def fix_length(wav: torch.Tensor, length: int, mode: str = "zero") -> torch.Tensor:
    """Truncate (from the start) or pad the last dim to `length`. mode: 'zero' | 'repeat'."""
    n = wav.shape[-1]
    if n >= length:
        return wav[..., :length]
    if mode == "zero":
        return torch.nn.functional.pad(wav, (0, length - n))
    if mode == "repeat":
        if n == 0:
            return torch.zeros(*wav.shape[:-1], length, dtype=wav.dtype)
        reps = -(-length // n)
        return wav.repeat(*([1] * (wav.dim() - 1)), reps)[..., :length]
    raise ValueError(f"unknown pad mode {mode}")


class _AudioListDataset(torch.utils.data.Dataset):

    def __init__(self, items: Sequence[Tuple[str, Path]], load_fn: Callable[[Path], torch.Tensor]):
        self.items = list(items)
        self.load_fn = load_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, path = self.items[i]
        try:
            return key, self.load_fn(path)
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")
            return key, None


def _list_collate(batch):
    return batch


def iter_audio_batches(items: Sequence[Tuple[str, Path]], load_fn: Callable[[Path], object],
                       batch_size: int, num_workers: int = 0) -> Iterator[List[Tuple[str, object]]]:
    """Yield lists of (key, load_fn(path)) with parallel decoding; failed files are skipped (logged).

    `load_fn` must be picklable (module-level function or functools.partial) when num_workers > 0.
    """
    loader = torch.utils.data.DataLoader(_AudioListDataset(items, load_fn), batch_size=batch_size,
                                         num_workers=num_workers, collate_fn=_list_collate)
    for batch in loader:
        batch = [(k, w) for k, w in batch if w is not None]
        if batch:
            yield batch


# ------------------------------------------------------------------------------------ feature cache
class FeatureCache:
    """Per-file feature cache persisted with torch.save, invalidated by (mtime, size) of the source file.

    Keys are resolved absolute paths, so caches stay valid when the same file is evaluated via different
    relative paths, and regenerated audio is re-extracted automatically.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, dict] = {}
        self.dirty = False
        if self.path.exists():
            self.data = torch.load(self.path, weights_only=False)

    @staticmethod
    def _sig(file: Path) -> Tuple[int, int]:
        st = Path(file).stat()
        return st.st_mtime_ns, st.st_size

    @staticmethod
    def _key(file: Path) -> str:
        return str(Path(file).resolve())

    def get(self, file: Path):
        entry = self.data.get(self._key(file))
        if entry is None or tuple(entry["sig"]) != self._sig(file):
            return None
        return entry["value"]

    def put(self, file: Path, value) -> None:
        self.data[self._key(file)] = {"sig": self._sig(file), "value": value}
        self.dirty = True

    def missing(self, files: Sequence[Path]) -> List[Path]:
        return [f for f in files if self.get(f) is None]

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            torch.save(self.data, tmp)
            tmp.replace(self.path)
            self.dirty = False
