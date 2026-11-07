"""Functional unit tests for HW3 (`cfg.py`, `generate.py`, `retrieve.py`, `eval.py`, `common.py`).

Runs completely offline without loading heavy checkpoints or running model inference.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from cfg import (
    APGState,
    CFG_MODES,
    CFG_SCHEDULES,
    apply_cfg,
    cfg,
    cfg_apg,
    cfg_rescaled,
    scheduled_guidance_scale,
)
from clap_model import CLAP_MAX_SAMPLES, CLAP_SR, embed_audio_files, load_clap_waveform
from common import (
    MUSICCAPS_AUDIO_DIR,
    TEST_CSV,
    FeatureCache,
    fix_length,
    list_audio_files,
    load_audio,
    load_captions,
)
from eval import (
    PASST_SAMPLES,
    PASST_SR,
    VGGISH_SR,
    compute_clap_score,
    compute_fd,
    compute_kl_passt,
    evaluate_features,
    extract_passt_features,
    extract_vggish_features,
    format_comparison_table,
    load_passt_waveform,
    load_vggish_waveform,
    pair_gt_and_pred_keys,
)
from generate import build_parser as build_gen_parser, format_ablation_name, generate, generate_dataset
from retrieve import build_results, rank_audio


def _write_sine_wav(path: Path, freq: float = 440.0, duration_s: float = 2.0, sr: int = 16000) -> None:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    wav = (0.5 * np.sin(2.0 * np.pi * freq * t) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, wav)


class TestDataAndCommon(unittest.TestCase):

    def test_repo_captions_and_musiccaps_exist(self):
        captions = load_captions(TEST_CSV)
        self.assertEqual(len(captions), 100)
        self.assertEqual(list(captions.keys())[0], "0")
        self.assertEqual(list(captions.keys())[-1], "99")
        if MUSICCAPS_AUDIO_DIR.exists():
            mc_files = list_audio_files(MUSICCAPS_AUDIO_DIR)
            self.assertGreater(len(mc_files), 5000)

    def test_audio_io_resample_and_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "0.wav"
            _write_sine_wav(wav_path, freq=440.0, duration_s=2.0, sr=16000)

            wav_32k = load_audio(wav_path, sr=32000, remove_dc=True, max_seconds=1.5)
            self.assertEqual(wav_32k.shape, (48000,))
            self.assertAlmostEqual(float(wav_32k.mean().abs()), 0.0, places=3)

            padded = fix_length(wav_32k, 64000, mode="zero")
            self.assertEqual(padded.shape, (64000,))
            self.assertEqual(float(padded[48000:].abs().max()), 0.0)

            rep = fix_length(wav_32k, 64000, mode="repeat")
            self.assertEqual(rep.shape, (64000,))
            self.assertTrue(torch.allclose(rep[:16000], rep[48000:64000]))

            # FeatureCache persistence & automatic invalidation on file modification
            cache_file = tmp_path / "cache.pt"
            cache = FeatureCache(cache_file)
            self.assertIsNone(cache.get(wav_path))
            val = torch.tensor([1.0, 2.0, 3.0])
            cache.put(wav_path, val)
            cache.save()

            cache2 = FeatureCache(cache_file)
            self.assertTrue(torch.equal(cache2.get(wav_path), val))

            # Modify file size -> cache invalidated
            _write_sine_wav(wav_path, freq=880.0, duration_s=3.0, sr=16000)
            self.assertIsNone(cache2.get(wav_path))


class TestCFGAblations(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.u = torch.randn(1, 64, 128)
        self.c = torch.randn(1, 64, 128)

    def test_standard_cfg_identities(self):
        self.assertTrue(torch.allclose(cfg(self.u, self.c, 0.0), self.u, atol=1e-6))
        self.assertTrue(torch.allclose(cfg(self.u, self.c, 1.0), self.c, atol=1e-6))
        w = 7.0
        expected = self.u + w * (self.c - self.u)
        self.assertTrue(torch.allclose(cfg(self.u, self.c, w), expected))
        self.assertTrue(torch.allclose(apply_cfg(self.u, self.c, w, mode="uncond"), self.u))
        self.assertTrue(torch.allclose(apply_cfg(self.u, self.c, w, mode="cond"), self.c))

    def test_rescaled_cfg_bounds_std(self):
        out_phi0 = cfg_rescaled(self.u, self.c, 7.0, rescale_phi=0.0)
        self.assertTrue(torch.allclose(out_phi0, cfg(self.u, self.c, 7.0)))

        out_phi1 = cfg_rescaled(self.u, self.c, 7.0, rescale_phi=1.0)
        self.assertAlmostEqual(float(out_phi1.std()), float(self.c.std()), places=5)

    def test_apg_projection_and_momentum(self):
        # When eta=1, norm_threshold=0, momentum=0, APG reduces to standard CFG
        out_eta1 = cfg_apg(self.u, self.c, 7.0, eta=1.0, norm_threshold=0.0, state=None)
        self.assertTrue(torch.allclose(out_eta1, cfg(self.u, self.c, 7.0), atol=1e-5))

        # When eta=0, the update (out - c) is strictly orthogonal to c (cosine similarity = 0)
        out_eta0 = cfg_apg(self.u, self.c, 7.0, eta=0.0, norm_threshold=0.0, state=None)
        update = out_eta0 - self.c
        cos = F.cosine_similarity(update.flatten(1), self.c.flatten(1), dim=-1)
        self.assertAlmostEqual(float(cos.abs().max()), 0.0, places=5)

        # Momentum state updates across steps
        state = APGState(momentum=-0.5)
        s1 = apply_cfg(self.u, self.c, 7.0, mode="apg", apg_state=state)
        s2 = apply_cfg(self.u, self.c, 7.0, mode="apg", apg_state=state)
        self.assertFalse(torch.allclose(s1, s2))

    def test_guidance_schedules(self):
        # Linear & cosine start at 1.0 and end at base_scale
        for sched in ("linear", "cosine"):
            self.assertAlmostEqual(scheduled_guidance_scale(7.0, 0, 100, sched), 1.0, places=6)
            self.assertAlmostEqual(scheduled_guidance_scale(7.0, 99, 100, sched), 7.0, places=6)
            mid = scheduled_guidance_scale(7.0, 50, 101, sched)
            self.assertAlmostEqual(mid, 4.0, places=5)

        # Interval schedule applies scale only inside [interval_min, interval_max]
        self.assertAlmostEqual(scheduled_guidance_scale(7.0, 5, 100, "interval", 0.2, 0.8), 1.0)
        self.assertAlmostEqual(scheduled_guidance_scale(7.0, 50, 100, "interval", 0.2, 0.8), 7.0)
        self.assertAlmostEqual(scheduled_guidance_scale(7.0, 95, 100, "interval", 0.2, 0.8), 1.0)

        for mode in CFG_MODES:
            for sched in CFG_SCHEDULES:
                res = apply_cfg(self.u, self.c, 5.0, mode=mode, schedule=sched, step_index=3, num_steps=10)
                self.assertEqual(res.shape, self.c.shape)

    def test_mock_generate_loop_and_ablation_dataset(self):
        class DummyConfig:
            in_channels = 8
            sample_size = 16
            audio_channels = 2
            sampling_rate = 16000

        class DummyScheduler:
            def __init__(self):
                self.timesteps = torch.tensor([3, 2, 1])

            def set_timesteps(self, n, device=None):
                self.timesteps = torch.arange(n, 0, -1)

            def scale_model_input(self, x, t):
                return x

            def step(self, noise_pred, t, latents, **kwargs):
                class Out:
                    prev_sample = latents - 0.1 * noise_pred

                return Out()

        class DummyVAE:
            config = DummyConfig()

            def decode(self, latents):
                class Out:
                    sample = torch.tanh(latents[:, :2, :]).repeat_interleave(1000, dim=-1)

                return Out()

        class DummyTransformer:
            config = DummyConfig()

            def __call__(self, x, t, **kwargs):
                # Return deterministic different uncond and cond halves
                b = x.shape[0] // 2
                u = torch.full_like(x[:b], 0.1)
                c = torch.full_like(x[b:], 0.3)
                return (torch.cat([u, c], dim=0),)

        class DummyPipe:
            _execution_device = torch.device("cpu")
            vae = DummyVAE()
            scheduler = DummyScheduler()
            rotary_embed_dim = 4
            transformer = DummyTransformer()

            def encode_prompt(self, prompt, device, do_classifier_free_guidance=False):
                val = 0.2 if prompt else -0.1
                return torch.full((1, 4, 8), val)

            def encode_duration(self, start, end, device, do_cfg, batch_size=1):
                return torch.zeros(1, 1, 8), torch.ones(1, 1, 8)

            def prepare_latents(self, b, c, t, dtype, device, generator, audio_channels=2):
                return torch.zeros((b, c, t), dtype=dtype)

            def prepare_extra_step_kwargs(self, generator, eta):
                return {}

            def progress_bar(self, iterable):
                return iterable

        import types

        fake_diffusers = types.ModuleType("diffusers")
        fake_emb = types.ModuleType("diffusers.models.embeddings")
        fake_emb.get_1d_rotary_pos_embed = lambda dim, seq_len, **kw: (torch.zeros(seq_len, dim), torch.ones(seq_len, dim))

        with patch.dict(sys.modules, {"diffusers": fake_diffusers, "diffusers.models.embeddings": fake_emb}):
            pipe = DummyPipe()
            out_s1 = generate(pipe, "piano", guidance_scale=1.0, num_inference_steps=4, audio_end_in_s=0.5)
            out_s7 = generate(pipe, "piano", guidance_scale=7.0, num_inference_steps=4, audio_end_in_s=0.5)
            out_apg = generate(
                pipe, "piano", guidance_scale=7.0, num_inference_steps=4, audio_end_in_s=0.5, cfg_mode="apg"
            )
            self.assertEqual(out_s1.shape, (2, 8000))
            self.assertFalse(torch.allclose(out_s1, out_s7))
            self.assertFalse(torch.allclose(out_s7, out_apg))

            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / format_ablation_name(7.0, "apg", "cosine")
                generate_dataset(
                    pipe,
                    {"0": "a calm piano solo", "1": "upbeat electronic drums"},
                    out_dir,
                    guidance_scale=7.0,
                    num_inference_steps=3,
                    audio_end_in_s=0.5,
                    cfg_mode="apg",
                    cfg_schedule="cosine",
                )
                self.assertTrue((out_dir / "0.mp3").exists())
                self.assertTrue((out_dir / "1.mp3").exists())
                self.assertTrue((out_dir / "config.json").exists())

        args = build_gen_parser().parse_args(
            ["--csv", "test.csv", "--scales", "1.0", "3.0", "7.0", "--cfg_mode", "apg", "--cfg_schedule", "cosine"]
        )
        self.assertEqual(args.scales, [1.0, 3.0, 7.0])
        self.assertEqual(args.cfg_mode, "apg")


class TestRetrievalAndEval(unittest.TestCase):

    def test_retrieval_ranking_and_outputs(self):
        # 3 queries, 4 candidate tracks in 8-D space
        audio_keys = ["yt_a", "yt_b", "yt_c", "yt_d"]
        audio_emb = F.normalize(torch.eye(4, 8), dim=-1)
        # Query 0 aligns with yt_c (index 2), Query 1 aligns with yt_a (index 0)
        text_emb = torch.stack([audio_emb[2], audio_emb[0]])

        scores, indices = rank_audio(text_emb, audio_emb, topk=2)
        self.assertEqual(indices[:, 0].tolist(), [2, 0])
        self.assertTrue(torch.allclose(scores[:, 0], torch.ones(2), atol=1e-5))

        mapping, rows = build_results(["0", "1"], audio_keys, scores, indices)
        self.assertEqual(mapping, {"0": "yt_c", "1": "yt_a"})
        self.assertEqual(len(rows), 4)

        mapping_ext, _ = build_results(["0", "1"], audio_keys, scores, indices, with_ext=True)
        self.assertEqual(mapping_ext, {"0": "yt_c.mp3", "1": "yt_a.mp3"})

    def test_fad_kl_clap_metrics_and_extractors(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=(64, 16))
        y = rng.normal(2.0, 1.5, size=(64, 16))

        # Identical distributions have ~0 Fréchet distance; shifted distribution has positive FD
        fd_same = compute_fd(x, x)
        fd_diff = compute_fd(x, y)
        self.assertAlmostEqual(fd_same, 0.0, places=5)
        self.assertGreater(fd_diff, 50.0)

        # KL(GT || pred) is 0 when logits match and > 0 when they differ
        logits_a = torch.randn(10, 527)
        logits_b = torch.randn(10, 527)
        kl_zero = compute_kl_passt(logits_a, logits_a)
        kl_pos = compute_kl_passt(logits_b, logits_a)
        self.assertAlmostEqual(kl_zero["kl_softmax"], 0.0, places=5)
        self.assertGreater(kl_pos["kl_softmax"], 0.1)

        # CLAP cosine similarity
        emb = F.normalize(torch.randn(10, 512), dim=-1)
        mean_sim, per_sample = compute_clap_score(emb, emb)
        self.assertAlmostEqual(mean_sim, 1.0, places=5)
        self.assertEqual(per_sample.shape, (10,))

    def test_end_to_end_extract_and_evaluate_with_mocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_dir = tmp_path / "gt"
            pred_dir = tmp_path / "pred_s7"
            gt_dir.mkdir()
            pred_dir.mkdir()

            # Create MusicCaps-style GT files (named by ytid) and prediction files (named 0, 1)
            _write_sine_wav(gt_dir / "yt_00.mp3", freq=440.0, duration_s=1.5, sr=16000)
            _write_sine_wav(gt_dir / "yt_01.mp3", freq=660.0, duration_s=1.5, sr=16000)
            _write_sine_wav(pred_dir / "0.mp3", freq=440.0, duration_s=1.5, sr=16000)
            _write_sine_wav(pred_dir / "1.mp3", freq=660.0, duration_s=1.5, sr=16000)

            gt_files = list_audio_files(gt_dir)
            pred_files = list_audio_files(pred_dir)
            retrieval_map = {"0": "yt_00.mp3", "1": "yt_01.mp3"}

            # Verify waveform loaders for VGGish (16k), PaSST (32k, padded to 320000), CLAP (48k, 480000)
            self.assertEqual(load_vggish_waveform(pred_files["0"], 1.0).shape, (VGGISH_SR,))
            self.assertEqual(load_passt_waveform(pred_files["0"], 1.0).shape, (PASST_SAMPLES,))
            self.assertEqual(load_clap_waveform(pred_files["0"], 1.0).shape, (CLAP_MAX_SAMPLES,))

            class DummyVGGish:
                def extract_one(self, wav):
                    torch.manual_seed(int(wav.abs().sum().item() * 100) % 1000)
                    return torch.randn(4, 128)

            class DummyPaSST:
                def extract_batch(self, wavs):
                    b = wavs.shape[0]
                    # Deterministic features derived from waveform energy
                    e = wavs.abs().mean(dim=-1, keepdim=True)
                    logits = e.repeat(1, 527) + torch.arange(527).float() * 0.001
                    embed = e.repeat(1, 768) + torch.arange(768).float() * 0.001
                    return logits, embed

            class DummyCLAP:
                def embed_waveforms(self, wavs):
                    b = wavs.shape[0]
                    return F.normalize(torch.ones(b, 64), dim=-1)

            v_gt = extract_vggish_features(DummyVGGish(), gt_files, audio_length=1.5)
            v_pr = extract_vggish_features(DummyVGGish(), pred_files, audio_length=1.5)
            p_gt_l, p_gt_e = extract_passt_features(DummyPaSST(), gt_files, audio_length=1.5)
            p_pr_l, p_pr_e = extract_passt_features(DummyPaSST(), pred_files, audio_length=1.5)
            c_pr = embed_audio_files(DummyCLAP(), pred_files, audio_length=1.5)
            c_txt = {"0": F.normalize(torch.ones(64), dim=-1), "1": F.normalize(torch.ones(64), dim=-1)}

            metrics = evaluate_features(
                pred_vggish=v_pr,
                gt_vggish=v_gt,
                pred_passt_logits=p_pr_l,
                gt_passt_logits=p_gt_l,
                pred_passt_embed=p_pr_e,
                gt_passt_embed=p_gt_e,
                pred_clap_audio=c_pr,
                text_clap_emb=c_txt,
                retrieval_map=retrieval_map,
            )
            self.assertIn("FD-VGG", metrics)
            self.assertIn("FD-PASST", metrics)
            self.assertIn("KL-PASST-softmax", metrics)
            self.assertIn("CLAP-Score", metrics)
            self.assertAlmostEqual(metrics["FD-VGG"], 0.0, places=4)
            self.assertAlmostEqual(metrics["KL-PASST-softmax"], 0.0, places=5)
            self.assertAlmostEqual(metrics["CLAP-Score"], 1.0, places=5)

            table = format_comparison_table({"cfg_s1": metrics, "cfg_s7": metrics})
            self.assertIn("FD-VGG", table)
            self.assertIn("KL-PASST-softmax", table)
            self.assertIn("CLAP-Score", table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
