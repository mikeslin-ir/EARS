"""
Load video and extract frames + audio for downstream event and VLM modules.
Uses OpenCV for video and librosa for audio (extracted from video or separate stream).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def load_video_capture(path: str | Path):
    """Open video with OpenCV. Returns cap; caller must release."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap) -> tuple[int, float, int, int]:
    """Return (num_frames, fps, width, height)."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return n, fps, w, h


def probe_fps(path: str | Path, default: float = 30.0) -> float:
    """Frame rate of a video, without decoding anything.

    Callers that need only the frame rate should use this rather than
    positionally unpacking :func:`get_video_info`: four call sites had bound
    ``fps`` to the third element (the width), silently turning every gap
    expressed in seconds into a gap roughly 20x too large, and differently so
    per dataset (640 on XD-Violence vs 320 on UCF-Crime).
    """
    cap = load_video_capture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return float(fps) if fps and fps > 0 else default


def iter_frames(
    path: str | Path,
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield RGB frames (H, W, 3) until end or max_frames."""
    cap = load_video_capture(path)
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            # BGR -> RGB
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def read_frames_at_indices(
    path: str | Path,
    indices: Sequence[int],
    *,
    grayscale: bool = False,
) -> dict[int, np.ndarray]:
    """Decode the requested frame indices in a single sequential pass.

    Seeking with ``CAP_PROP_POS_FRAMES`` makes the decoder restart from the
    nearest keyframe on every call, which costs about 25x more than walking the
    stream once and keeping the frames we want -- 4.30s vs 0.17s per video on
    the 24-video benchmark in ``results/motion_proxy_cost.json``. The frames
    themselves are unaffected: seeking was verified pixel-identical to this pass
    on 3261/3261 sampled frames, so switching does not invalidate cached
    signals.

    Prefer seeking only when the indices are few and far apart, since this walks
    the stream up to ``max(indices)`` regardless of how many frames are kept.

    Returns a dict keyed by index; missing keys mean the read failed there.
    """
    wanted = sorted({int(i) for i in indices if i >= 0})
    if not wanted:
        return {}
    last = wanted[-1]
    todo = set(wanted)

    cap = load_video_capture(path)
    out: dict[int, np.ndarray] = {}
    try:
        i = 0
        while i <= last:
            if not cap.grab():
                break
            if i in todo:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    out[i] = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale
                              else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
    finally:
        cap.release()
    return out


def extract_audio_from_video(path: str | Path, sr: int = 22050) -> np.ndarray:
    """
    Extract audio from video file. Uses ffmpeg to decode then librosa to load.
    Returns 1D waveform at sample rate sr.
    If ffmpeg is missing or extraction fails, returns zeros and logs a warning (install ffmpeg for real audio).
    """
    import subprocess
    import tempfile
    import warnings
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", "-f", "wav", wav_path,
        ], capture_output=True, timeout=60)
        if r.returncode != 0 or not Path(wav_path).exists():
            if not getattr(extract_audio_from_video, "_warned_ffmpeg_fail", False):
                warnings.warn(
                    "Audio extraction failed (ffmpeg error or no audio track). Using silence. Install ffmpeg and ensure videos have audio for non-flat E_t.",
                    UserWarning,
                    stacklevel=2,
                )
                extract_audio_from_video._warned_ffmpeg_fail = True
            return np.zeros(sr * 10, dtype=np.float32)
        if not LIBROSA_AVAILABLE:
            return np.zeros(sr * 10, dtype=np.float32)
        y, _ = librosa.load(wav_path, sr=sr, mono=True)
        return y
    except FileNotFoundError:
        if not getattr(extract_audio_from_video, "_warned_ffmpeg", False):
            warnings.warn(
                "ffmpeg not found. Audio will be silent. Install with: sudo apt install ffmpeg (or conda install ffmpeg).",
                UserWarning,
                stacklevel=2,
            )
            extract_audio_from_video._warned_ffmpeg = True
        return np.zeros(sr * 10, dtype=np.float32)
    except (subprocess.TimeoutExpired, Exception):
        if not getattr(extract_audio_from_video, "_warned_audio_fail", False):
            warnings.warn(
                "Audio extraction failed (timeout or error). Using silence.",
                UserWarning,
                stacklevel=2,
            )
            extract_audio_from_video._warned_audio_fail = True
        return np.zeros(sr * 10, dtype=np.float32)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def frames_to_time_indices(frame_indices: list[int], fps: float) -> list[float]:
    """Convert frame indices to time in seconds."""
    return [i / fps for i in frame_indices]


def load_frame_at_index(path: str | Path, frame_index: int) -> np.ndarray | None:
    """Load a single frame by index (0-based). Returns RGB (H,W,3) or None."""
    cap = load_video_capture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
