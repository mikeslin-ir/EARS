"""Public Python API for EARS frame selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .selection import select_peaks
from .signals import audio_signal, evidence_scores, motion_signal
from .video import probe_video, read_frames


@dataclass(frozen=True)
class EARSConfig:
    motion_candidates: int = 256
    flow_scale: float = 0.25
    audio_sample_rate: int = 22_050
    audio_segment_seconds: float = 1.0
    audio_weight: float = 1.0
    motion_weight: float = 1.0
    min_gap_seconds: float = 0.5

    def validate(self) -> None:
        if self.motion_candidates < 2:
            raise ValueError("motion_candidates must be at least two")
        if not 0 < self.flow_scale <= 1:
            raise ValueError("flow_scale must be in (0, 1]")
        if self.audio_sample_rate <= 0 or self.audio_segment_seconds <= 0:
            raise ValueError("Audio settings must be positive")
        if min(self.audio_weight, self.motion_weight, self.min_gap_seconds) < 0:
            raise ValueError("Weights and min_gap_seconds must be nonnegative")


@dataclass(frozen=True)
class SelectionResult:
    video_path: str
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    scores: tuple[float, ...]
    fps: float
    num_frames: int
    frames: tuple[np.ndarray, ...] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "video_path": self.video_path,
            "frame_indices": list(self.frame_indices),
            "timestamps": list(self.timestamps),
            "scores": list(self.scores),
            "fps": self.fps,
            "num_frames": self.num_frames,
        }

    def save_frames(self, directory: str | Path, *, prefix: str = "frame") -> list[Path]:
        if self.frames is None:
            raise ValueError("Frames were not returned; call select(..., return_frames=True)")
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, frame in zip(self.frame_indices, self.frames, strict=True):
            path = output / f"{prefix}_{index:08d}.jpg"
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write frame: {path}")
            paths.append(path)
        return paths


class EARSSelector:
    def __init__(self, config: EARSConfig | None = None) -> None:
        self.config = config or EARSConfig()
        self.config.validate()

    def select(
        self,
        video_path: str | Path,
        k: int,
        *,
        return_frames: bool = False,
    ) -> SelectionResult:
        if k <= 0:
            raise ValueError("k must be positive")
        path = Path(video_path)
        info = probe_video(path)
        motion = motion_signal(
            path,
            info,
            candidates=self.config.motion_candidates,
            flow_scale=self.config.flow_scale,
        )
        audio = audio_signal(
            path,
            info,
            sample_rate=self.config.audio_sample_rate,
            segment_seconds=self.config.audio_segment_seconds,
        )
        scores = evidence_scores(
            audio,
            motion,
            audio_weight=self.config.audio_weight,
            motion_weight=self.config.motion_weight,
        )
        gap = round(self.config.min_gap_seconds * info.fps)
        selected = select_peaks(scores, k, gap)
        decoded = read_frames(path, selected) if return_frames else None
        if decoded is not None and set(decoded) != set(selected.tolist()):
            raise RuntimeError("Failed to decode one or more selected frames")
        return SelectionResult(
            video_path=str(path),
            frame_indices=tuple(int(index) for index in selected),
            timestamps=tuple(float(index / info.fps) for index in selected),
            scores=tuple(float(scores[index]) for index in selected),
            fps=info.fps,
            num_frames=info.num_frames,
            frames=(
                tuple(decoded[int(index)] for index in selected)
                if decoded is not None
                else None
            ),
        )
