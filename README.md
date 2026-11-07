# DeepMIR HW3: Text-to-Music Retrieval, Generation (CFG Ablation), and Evaluation

This repository implements the full pipeline for **Homework 3 (Text-to-Music Generation and Evaluation)**:
1. **Text-to-Music Retrieval (`code/retrieve.py`)**: Finds the closest reference audio track in **MusicCaps** (`data/audio_mp3/`) for each of the 100 test prompts in `test.csv` using **LAION-CLAP** (`music_audioset_epoch_15_esc_90.14.pt`).
2. **Music Generation & CFG Ablation Study (`code/cfg.py`, `code/generate.py`)**: Implements classifier-free guidance (CFG) for **Stable Audio Open (`stabilityai/stable-audio-open-1.0`)**, along with guidance-mode and timestep-schedule ablations (`standard`, `uncond`, `cond`, `rescaled`, and **Adaptive Projected Guidance (`apg`)**).
3. **Evaluation (`code/eval.py`)**: Computes **FAD (`FD-VGG`, `FD-PASST`)**, **KL Divergence (`KL-PASST-softmax`)**, and **CLAP Score (`CLAP-Score`)** based on `third_party/av-benchmark`.

---

## Repository Structure

```text
deepmir-hw3/
├── code/
│   ├── cfg.py            # Standard CFG + ablation variants (Rescaled CFG, APG, timestep schedules)
│   ├── generate.py       # Stable Audio Open single-prompt & batch/ablation dataset generation
│   ├── retrieve.py       # LAION-CLAP retrieval over MusicCaps (outputs retrieval.json & top-k CSV)
│   ├── eval.py           # FAD (FD-VGG, FD-PaSST), KL-PaSST, and CLAP-Score evaluation
│   ├── clap_model.py     # LAION-CLAP wrapper (auto-downloads music_audioset_epoch_15_esc_90.14.pt)
│   ├── common.py         # Audio I/O, resampling, caption loading, and mtime-aware FeatureCache
│   └── test.py           # Offline functional and unit test suite
├── data/
│   └── audio_mp3/        # MusicCaps reference audio tracks (<ytid>.mp3)
├── third_party/
│   └── av-benchmark/     # Cloned hkchengrex/av-benchmark evaluation library
├── test.csv              # 100 test prompts (id, caption)
├── pyproject.toml        # Project dependencies & optional [eval] group for uv
├── uv.lock               # Locked dependencies for uv
├── requirements.txt      # Exported requirements list
└── README.md             # This file
```

---

## Environment Setup (`uv`)

### 1. Create Virtual Environment & Sync Dependencies

This repository uses **[uv](https://docs.astral.sh/uv/)** for package and environment management (Python 3.11+, ~9 GB VRAM in fp16 for Stable Audio Open).

```bash
# Sync core + evaluation dependencies (including laion-clap and passt) into .venv
uv sync --extra eval

# (Optional) If you need a specific CUDA wheel index for torch/torchaudio (e.g., cu128):
uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Ensure `third_party/av-benchmark` is present:

```bash
# If third_party/av-benchmark is not yet cloned:
git clone https://github.com/hkchengrex/av-benchmark.git third_party/av-benchmark
```

*(Note: `code/eval.py` imports `av_bench` directly from `third_party/av-benchmark` and only loads VGGish, PaSST, and LAION-CLAP, so `ImageBind` and `Synchformer` checkpoints are **not** required.)*

### 2. Hugging Face Authentication & Pretrained Weights

`stabilityai/stable-audio-open-1.0` is a gated model on Hugging Face:
1. Accept the license at [https://huggingface.co/stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0).
2. Set your token via `uv run huggingface-cli login` or create a `.env` file in the repo root:
   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN=hf_...
   ```

The HW3-required LAION-CLAP checkpoint (`music_audioset_epoch_15_esc_90.14.pt`) is **automatically downloaded** from `lukewys/laion_clap` into `weights/music_audioset_epoch_15_esc_90.14.pt` on first run of `retrieve.py` or `eval.py`. You can also download it manually:

```bash
mkdir -p weights
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt \
     -O weights/music_audioset_epoch_15_esc_90.14.pt
```

---

## Step-by-Step Usage

### Task 1: MusicCaps Retrieval (`code/retrieve.py`)

Find the closest MusicCaps reference track in `data/audio_mp3/` for each prompt in `test.csv` using LAION-CLAP cosine similarity:

```bash
uv run python code/retrieve.py \
    --captions test.csv \
    --audio_dir data/audio_mp3 \
    --out_dir outputs/retrieval \
    --topk 5 \
    --batch_size 32 \
    --num_workers 4
```

*Add `--with_ext` if you want the values in `retrieval.json` to include `.mp3` (e.g. `"0": "abc123.mp3"` instead of `"0": "abc123"`).*

**Outputs:**
- `outputs/retrieval/retrieval.json`: Dictionary mapping `"0"`..`"99"` (string keys) to the top-1 MusicCaps ID (string values) for submission.
- `outputs/retrieval/retrieval_topk.csv`: Top-$k$ retrieved tracks and cosine similarity scores per prompt for report inspection.
- `outputs/cache/clap_audio_music_audioset_epoch_15_esc_90.14.pt`: Cached MusicCaps CLAP embeddings (reused automatically by `eval.py`).

---

### Task 2: Music Generation & CFG Ablation Study (`code/generate.py`)

#### Verify Standard CFG Implementation
Compare `code/cfg.py` against `diffusers.StableAudioPipeline` on a single prompt:

```bash
uv run python code/generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.mp3 --check
```

#### Generate Full Test Set (100 Prompts)
By default, `code/generate.py` saves `.mp3` files (`0.mp3`..`99.mp3`), matching the NTU COOL submission requirement and the MusicCaps `.mp3` dataset:

```bash
# Baseline generation (standard CFG, guidance_scale = 7.0, 100 steps, 10.0s, outputs 0.mp3..99.mp3)
uv run python code/generate.py \
    --csv test.csv \
    --out_dir outputs/gen/cfg_standard_s7 \
    --scale 7.0 \
    --steps 100 \
    --seconds 10.0
```

#### Run CFG Ablation Sweeps
`generate.py` treats CFG as a modular ablation component (`code/cfg.py`):
- **Guidance Scale Sweep (`--scales`)**: Loads the pipeline once and generates a subfolder per scale (`cfg_<mode>_s<scale>`).
  ```bash
  uv run python code/generate.py \
      --csv test.csv \
      --out_dir outputs/gen \
      --scales 1.0 3.0 5.0 7.0 10.0 15.0
  ```
- **Guidance Mode Ablation (`--cfg_mode`)**:
  - `standard`: $\hat{v} = v_{\text{uncond}} + w (v_{\text{cond}} - v_{\text{uncond}})$
  - `uncond`: unconditional generation ($w = 0$)
  - `cond`: conditional generation without guidance ($w = 1$)
  - `rescaled`: Rescaled CFG (Lin et al., 2024) matching conditional standard deviation with factor `--rescale_phi` (default `0.7`).
  - `apg`: **Adaptive Projected Guidance** (Sadat et al., ICLR 2025), decomposing $v_{\text{cond}} - v_{\text{uncond}}$ into parallel and orthogonal components relative to $v_{\text{cond}}$ to eliminate oversaturation/artifacts at high guidance scales:
    ```bash
    uv run python code/generate.py \
        --csv test.csv \
        --out_dir outputs/gen \
        --cfg_mode apg \
        --scales 7.0 10.0 15.0 \
        --apg_eta 0.0 \
        --apg_momentum -0.5
    ```
- **Timestep Schedule Ablation (`--cfg_schedule`)**:
  - `constant` (default), `linear` (ramp from $1 \to w$), `cosine` (smooth cosine ramp from $1 \to w$), or `interval` (apply $w$ only when normalized step progress $s \in [\text{interval\_min}, \text{interval\_max}]$):
    ```bash
    uv run python code/generate.py \
        --csv test.csv \
        --out_dir outputs/gen \
        --cfg_mode standard \
        --cfg_schedule interval --interval_min 0.2 --interval_max 0.8 \
        --scales 7.0
    ```
- **Negative Prompting (`--negative_prompt`)**: Replaces the default zero unconditional embedding with an encoded negative text prompt:
  ```bash
  uv run python code/generate.py \
      --csv test.csv \
      --out_dir outputs/gen/cfg_neg_s7 \
      --scale 7.0 \
      --negative_prompt "low quality, noise, distortion, static, muffled"
  ```

---

### Task 3: Evaluation (`code/eval.py`)

Evaluate one or more generated audio directories on **FAD (`FD-VGG`, `FD-PASST`)**, **KL (`KL-PASST-softmax`)**, and **CLAP (`CLAP-Score`)**:

```bash
# Evaluate a single run (uses retrieved MusicCaps tracks for paired KL and GT reference by default)
uv run python code/eval.py \
    --pred_dir outputs/gen/cfg_standard_s7 \
    --gt_dir data/audio_mp3 \
    --captions test.csv \
    --retrieval_json outputs/retrieval/retrieval.json

# Evaluate all CFG ablation runs at once and save a comparison table (JSON + CSV)
uv run python code/eval.py \
    --pred_dir outputs/gen/cfg_* \
    --gt_dir data/audio_mp3 \
    --captions test.csv \
    --retrieval_json outputs/retrieval/retrieval.json \
    --gt_subset retrieved \
    --out outputs/eval/ablation_summary.json
```

**Metric Details & Options:**
| Metric | Key in Output | Better | Description |
| :--- | :--- | :---: | :--- |
| **FAD (VGGish)** | `FD-VGG` | $\downarrow$ | Fréchet Distance on 128-D `torchvggish` embeddings (16 kHz, 0.96 s patches). |
| **FAD (PaSST)** | `FD-PASST` | $\downarrow$ | Fréchet Distance on 768-D `hear21passt` embeddings (32 kHz, 10 s clips). |
| **KL (PaSST)** | `KL-PASST-softmax` | $\downarrow$ | Paired $\mathrm{KL}(p_{\text{GT}} \parallel p_{\text{pred}})$ over 527 AudioSet logits paired via `retrieval.json`. |
| **CLAP Score** | `CLAP-Score` | $\uparrow$ | Mean cosine similarity between test prompts and generated audio (`music_audioset_epoch_15_esc_90.14.pt`). |

- `--gt_subset {retrieved,all}`: By default (`retrieved`), extracts VGGish/PaSST features for the 100 retrieved MusicCaps reference clips (fast). Pass `--gt_subset all` to compute FAD against all 5,367 MusicCaps tracks.
- All extracted GT and prediction features are cached in `outputs/cache/` and automatically invalidated if an audio file is regenerated.

---

## Running Unit & Functional Tests

To verify `cfg.py`, `generate.py`, `retrieve.py`, `eval.py`, and `common.py` offline without running heavy model inference:

```bash
uv run python code/test.py
```

## References & Citations

- **Stable Audio Open**: Evans et al., *"Stable Audio Open"*, 2024 (`stabilityai/stable-audio-open-1.0`).
- **av-benchmark / MMAudio**: Cheng et al., *"MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis"*, CVPR 2025 ([hkchengrex/av-benchmark](https://github.com/hkchengrex/av-benchmark)).
- **LAION-CLAP**: Wu et al., *"Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation"*, ICASSP 2023 ([LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)).
- **PaSST**: Koutini et al., *"Efficient Training of Audio Transformers with Patchout"*, Interspeech 2022 ([kkoutini/passt_hear21](https://github.com/kkoutini/passt_hear21)).
- **FAD**: Kilgour et al., *"Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms"*, Interspeech 2019.
- **Adaptive Projected Guidance (APG)**: Sadat et al., *"Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models"*, ICLR 2025.
- **Rescaled CFG**: Lin et al., *"Common Diffusion Noise Schedules and Sample Steps are Flawed"*, WACV 2024.
