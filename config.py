"""Portable defaults for EARS experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


XD_VIOLENCE_ROOT = Path(os.environ.get("XD_VIOLENCE_ROOT", "datasets/XD_violence"))
UCF_CRIME_ROOT = Path(os.environ.get("UCF_CRIME_ROOT", "datasets/UCF-Crime"))


@dataclass
class EventAgentConfig:
    alpha: float = 1.0
    beta: float = 1.0
    top_k: int = 8
    tau_sec: float = 0.5


@dataclass
class RetrievalConfig:
    num_frames: int = 8


@dataclass
class VLMConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_new_tokens: int = 64
    device: str = "cuda"


VLM_PROMPT = (
    "These frames were selected because abnormal sound and motion were detected. "
    "Does the scene indicate a violent or dangerous event? "
    "Answer with one word: Yes or No."
)

TRAIN_FRACTIONS = (1.0, 0.2, 0.1, 0.05)