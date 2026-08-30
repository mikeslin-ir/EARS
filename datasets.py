"""
Dataset loaders for XD-Violence and UCF-Crime.
Returns lists of (video_path, label) where label is True = anomaly/violence, False = normal.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator

from config import XD_VIOLENCE_ROOT, UCF_CRIME_ROOT


def iter_xd_violence(
    root: Path | None = None,
    split: str = "test",
) -> Iterator[tuple[Path, bool]]:
    """
    XD-Violence: root/train and root/test each have subdirs Fighting, Normal, Shooting.
    Yield (video_path, is_violence). Normal -> False, Fighting/Shooting -> True.
    """
    root = root or XD_VIOLENCE_ROOT
    split_dir = root / split
    if not split_dir.exists():
        return
    for sub in ("Fighting", "Shooting", "Normal"):
        d = split_dir / sub
        if not d.is_dir():
            continue
        is_violence = sub != "Normal"
        for p in d.iterdir():
            if p.suffix.lower() in (".mp4", ".avi", ".mkv"):
                yield (p, is_violence)


def list_xd_violence(
    root: Path | None = None,
    split: str = "test",
    fraction: float = 1.0,
) -> list[tuple[Path, bool]]:
    """List all (path, label) for split; fraction in (0,1] for small-data."""
    items = list(iter_xd_violence(root, split))
    if fraction < 1.0 and items:
        n = max(1, int(len(items) * fraction))
        items = items[:n]
    return items


def _ucf_find_video_path(video_name: str, root: Path) -> Path | None:
    """Resolve e.g. Abuse001_x264.mp4 or Abuse/Abuse001_x264.mp4 to full path."""
    name = Path(video_name).name
    for part in (
        "Anomaly-Videos-Part-1", "Anomaly-Videos-Part-2",
        "Anomaly-Videos-Part-3", "Anomaly-Videos-Part-4",
        "Training-Normal-Videos-Part-1", "Training-Normal-Videos-Part-2",
        "Testing_Normal_Videos_Anomaly", "Normal_Videos_for_Event_Recognition",
    ):
        d = root / part
        if not d.is_dir():
            continue
        for f in d.rglob(name):
            if f.is_file():
                return f
    return None


def load_ucf_temporal_annotations(root: Path | None = None) -> list[tuple[str, str, list[tuple[int, int]]]]:
    """
    Parse Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
    Returns list of (video_name, class_name, [(start_frame, end_frame), ...]).
    """
    root = root or UCF_CRIME_ROOT
    ann_path = root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    if not ann_path.exists():
        return []
    out = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name, cls = parts[0], parts[1]
            frames = []
            i = 2
            while i + 1 < len(parts):
                a, b = int(parts[i]), int(parts[i + 1])
                if a >= 0 and b >= 0:
                    frames.append((a, b))
                i += 2
            out.append((name, cls, frames))
    return out


def iter_ucf_crime_test(
    root: Path | None = None,
) -> Iterator[tuple[Path, bool]]:
    """
    UCF-Crime test: use temporal annotation file; each video is anomaly (True).
    Resolve path from Part folders. Yields (video_path, True) for each test anomaly video.
    """
    root = root or UCF_CRIME_ROOT
    for name, _cls, _ranges in load_ucf_temporal_annotations(root):
        path = _ucf_find_video_path(name, root)
        if path is not None:
            yield (path, True)
    # Optionally add normal test videos from Testing_Normal_Videos_Anomaly
    normal_dir = root / "Testing_Normal_Videos_Anomaly"
    if normal_dir.exists():
        for f in normal_dir.rglob("*.mp4"):
            if f.is_file():
                yield (f, False)


def list_ucf_crime_test(
    root: Path | None = None,
    fraction: float = 1.0,
) -> list[tuple[Path, bool]]:
    """List test (path, label); fraction for small-data."""
    items = list(iter_ucf_crime_test(root))
    if fraction < 1.0 and items:
        n = max(1, int(len(items) * fraction))
        items = items[:n]
    return items
