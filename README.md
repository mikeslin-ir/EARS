# EARS

EARS is a lightweight Python API for selecting evidence-rich frames from a
video using motion and audio activity. It is training-free, query-free, and
does not require a vision-language model.

## Installation

EARS requires Python 3.10+ and FFmpeg.

```bash
sudo apt-get install ffmpeg
git clone https://github.com/mikeslin-ir/EARS.git
cd EARS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Python API

```python
from ears import EARSSelector

selector = EARSSelector()
result = selector.select("video.mp4", k=4, return_frames=True)

print(result.frame_indices)
print(result.timestamps)
print(result.scores)

# Save the returned RGB NumPy arrays as JPEG files.
result.save_frames("selected_frames")
```

`SelectionResult` contains:

- `frame_indices`: selected zero-based frame indices;
- `timestamps`: selected timestamps in seconds;
- `scores`: EARS evidence scores at the selected frames;
- `fps` and `num_frames`: decoded video metadata;
- `frames`: optional RGB NumPy arrays when `return_frames=True`.

Configuration is explicit:

```python
from ears import EARSConfig, EARSSelector

config = EARSConfig(
    motion_candidates=256,
    flow_scale=0.25,
    audio_sample_rate=22_050,
    audio_segment_seconds=1.0,
    audio_weight=1.0,
    motion_weight=1.0,
    min_gap_seconds=0.5,
)
selector = EARSSelector(config)
```

See [API documentation](docs/API.md) for the complete contract.

## Command Line

The installed command writes the selection as JSON:

```bash
ears-select video.mp4 -k 4 \
  --json outputs/selection.json \
  --frames-dir outputs/frames
```

Equivalent Python and bash entry points are included:

```bash
python scripts/select_frames.py video.mp4 -k 4
bash scripts/select_frames.sh video.mp4 -k 4
```

See [CLI documentation](docs/CLI.md) for all options.

## How It Works

1. Sample up to 256 endpoint-inclusive motion candidates.
2. Compute quarter-scale Farneback optical-flow displacement.
3. Decode mono audio and compute one-second mean absolute amplitude.
4. Interpolate both signals onto the native frame grid.
5. Normalize and fuse the signals with a sigmoid score.
6. Select temporal peaks and backfill deterministically to return exactly
   `min(k, num_frames)` frames.

Videos without an audio stream are treated as silent. Missing FFmpeg,
decoding failures, invalid metadata, and incomplete frame reads raise explicit
errors.

## Repository Layout

```text
EARS/
├── src/ears/               # installable Python package
│   ├── api.py              # EARSSelector and result types
│   ├── signals.py          # audio and motion evidence
│   ├── selection.py        # temporal peak selection
│   ├── video.py            # strict decoding utilities
│   └── cli.py              # command-line interface
├── scripts/
│   ├── select_frames.py
│   └── select_frames.sh
├── examples/
│   └── python_api.py
├── tests/
├── docs/
└── data/                   # optional local inputs; ignored by Git
```

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
