"""Command-line interface for EARS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import EARSConfig, EARSSelector


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="ears-select",
        description="Select evidence-rich frames from a video.",
    )
    command.add_argument("video", type=Path)
    command.add_argument("-k", type=int, required=True, help="Number of frames to select")
    command.add_argument("--json", type=Path, dest="json_path")
    command.add_argument("--frames-dir", type=Path)
    command.add_argument("--motion-candidates", type=int, default=256)
    command.add_argument("--flow-scale", type=float, default=0.25)
    command.add_argument("--audio-weight", type=float, default=1.0)
    command.add_argument("--motion-weight", type=float, default=1.0)
    command.add_argument("--min-gap-seconds", type=float, default=0.5)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = EARSConfig(
        motion_candidates=args.motion_candidates,
        flow_scale=args.flow_scale,
        audio_weight=args.audio_weight,
        motion_weight=args.motion_weight,
        min_gap_seconds=args.min_gap_seconds,
    )
    result = EARSSelector(config).select(
        args.video,
        args.k,
        return_frames=args.frames_dir is not None,
    )
    payload = result.to_dict()
    if args.frames_dir is not None:
        payload["saved_frames"] = [
            str(path) for path in result.save_frames(args.frames_dir)
        ]
    rendered = json.dumps(payload, indent=2)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
