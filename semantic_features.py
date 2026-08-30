#!/usr/bin/env python3
"""SigLIP-based semantic frame-text matching signal for EARS.

Provides a *question-conditional* signal S_t = max_i cos(SigLIP_img(f_t),
SigLIP_txt(query_i)) that complements the audio-energy E_t and motion M_t
signals. Unlike E_t/M_t, S_t depends on the question text and therefore
cannot be cached video-level — what we cache is the (T, D) per-frame image
embedding tensor (question-independent), and S_t is computed at query time
by encoding the candidate option strings and taking max-over-options.

Frame alignment: per-video frame embeddings are computed on T frames spaced
at the same MOTION_STRIDE used by motion_features, so the resulting tensor
is index-aligned to the cached M_t (and the upsampled E_t).

API
---
- SemanticEncoder: wraps a SigLIP model + processor.
- compute_video_frame_embeddings(path, T, encoder) -> (T, D) float32.
- build_semantic_cache(dataset, encoder, cache_dir, target_T_lookup)
  -> dict[Path, np.ndarray].
- compute_semantic_signal(frame_emb, text_emb, mode="max") -> (T,) float32.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@dataclass
class SemanticConfig:
    model_id: str = "google/siglip-base-patch16-224"
    batch_size: int = 32
    device: str = "cuda"
    # 0.0 = no L2 normalize (raw SigLIP); siglip uses sigmoid loss so cosine
    # similarity in [-1, 1] is fine after L2 normalize.
    l2_normalize: bool = True


class SemanticEncoder:
    """Lazy-loaded SigLIP image+text encoder."""

    def __init__(self, cfg: SemanticConfig | None = None):
        self.cfg = cfg or SemanticConfig()
        self.model = None
        self.processor = None
        self._device = None
        self._dtype = torch.float32

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self.model = AutoModel.from_pretrained(self.cfg.model_id)
        self._device = torch.device(self.cfg.device if torch.cuda.is_available()
                                    else "cpu")
        if self._device.type == "cuda":
            self._dtype = torch.float16
            self.model = self.model.to(self._device, dtype=self._dtype)
        else:
            self.model = self.model.to(self._device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode a list of HxWx3 uint8 RGB arrays. Returns (N, D) float32."""
        if not frames:
            return np.zeros((0, 768), dtype=np.float32)
        self.load()
        out: list[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in range(0, len(frames), bs):
            batch = frames[i:i + bs]
            pil = [Image.fromarray(f) for f in batch]
            inputs = self.processor(images=pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device, dtype=self._dtype)
            feats = self.model.get_image_features(pixel_values=pixel_values)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]
            if self.cfg.l2_normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode N strings. Returns (N, D) float32."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        self.load()
        # SigLIP requires padding="max_length" (its tokenizer is fixed-len).
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding="max_length", truncation=True)
        input_ids = inputs["input_ids"].to(self._device)
        feats = self.model.get_text_features(input_ids=input_ids)
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        if self.cfg.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats.float().cpu().numpy()


def _cache_path(cache_dir: Path, video_path: Path, model_id: str) -> Path:
    h = hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:16]
    mh = hashlib.sha1(model_id.encode()).hexdigest()[:8]
    return cache_dir / f"{video_path.stem}_{h}_{mh}.npz"


def compute_video_frame_embeddings(
    video_path: Path,
    T: int,
    encoder: SemanticEncoder,
    motion_stride: int = 5,
) -> np.ndarray:
    """Extract T frames at motion-stride-aligned positions, encode with SigLIP.

    The first frame is at index 0 (matching motion_magnitude_downsampled),
    subsequent frames at index t * motion_stride.
    """
    from video_loader import load_video_capture, get_video_info, load_frame_at_index

    cap = load_video_capture(video_path)
    try:
        total_frames, _, _, _ = get_video_info(cap)
    finally:
        cap.release()
    if total_frames <= 0 or T <= 0:
        return np.zeros((0, 768), dtype=np.float32)

    indices = [min(t * motion_stride, total_frames - 1) for t in range(T)]
    frames: list[np.ndarray] = []
    for fi in indices:
        f = load_frame_at_index(video_path, fi)
        if f is not None:
            frames.append(f)
        else:
            # On read failure pad with a black frame so length stays aligned.
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return encoder.encode_images(frames)


def build_semantic_cache(
    dataset: list,
    encoder: SemanticEncoder,
    cache_dir: Path,
    target_T_lookup: dict,
    motion_stride: int = 5,
) -> dict:
    """Pre-compute and disk-cache per-video frame embeddings.

    Args
    ----
    dataset: list whose items are tuples whose first element is a Path.
    encoder: loaded SemanticEncoder.
    cache_dir: directory for `.npz` cache files.
    target_T_lookup: dict mapping video Path -> target T (matches len(M_t)).
    motion_stride: must match motion_features stride used elsewhere.

    Returns
    -------
    dict[Path, np.ndarray] of shape (T, D) per video.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[Path, np.ndarray] = {}
    for item in tqdm(dataset, desc="Precomputing SigLIP frame embeddings",
                     unit="video"):
        vp = item[0] if isinstance(item, tuple) else item
        T = int(target_T_lookup.get(vp, 0))
        if T <= 0:
            continue
        cf = _cache_path(cache_dir, vp, encoder.cfg.model_id)
        if cf.exists():
            try:
                z = np.load(cf)
                if z["frame_emb"].shape[0] == T:
                    out[vp] = z["frame_emb"].astype(np.float32)
                    continue
            except Exception:
                pass
        emb = compute_video_frame_embeddings(vp, T, encoder, motion_stride)
        if emb.shape[0] == T:
            out[vp] = emb
            try:
                np.savez_compressed(cf, frame_emb=emb)
            except Exception:
                pass
    return out


def compute_semantic_signal(
    frame_emb: np.ndarray,
    text_emb: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """Question-conditional frame relevance signal.

    Args
    ----
    frame_emb: (T, D) per-frame embedding.
    text_emb:  (N, D) per-query embedding (e.g., one per option, L2 normalized).
    mode:      'max' (default), 'mean', or 'sum'.

    Returns
    -------
    S_t: (T,) float32, one score per frame.
    """
    if frame_emb.size == 0 or text_emb.size == 0:
        return np.zeros((frame_emb.shape[0],), dtype=np.float32)
    sims = frame_emb @ text_emb.T  # (T, N)
    if mode == "max":
        return sims.max(axis=1).astype(np.float32)
    if mode == "mean":
        return sims.mean(axis=1).astype(np.float32)
    if mode == "sum":
        return sims.sum(axis=1).astype(np.float32)
    raise ValueError(f"unknown mode: {mode}")
