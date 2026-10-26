# Python API

## `EARSSelector`

```python
from ears import EARSSelector

result = EARSSelector().select(
    "video.mp4",
    k=4,
    return_frames=False,
)
```

### Parameters

- `video_path`: path to a readable video.
- `k`: positive number of frames to return. Videos shorter than `k` return one
  result per available frame.
- `return_frames`: decode selected frames as RGB NumPy arrays.

### Result

`SelectionResult` exposes:

```python
result.frame_indices  # tuple[int, ...]
result.timestamps     # tuple[float, ...]
result.scores         # tuple[float, ...]
result.fps            # float
result.num_frames     # int
result.frames         # tuple[np.ndarray, ...] | None
result.to_dict()      # JSON-compatible metadata
```

To save returned arrays:

```python
result = EARSSelector().select("video.mp4", 4, return_frames=True)
paths = result.save_frames("selected_frames", prefix="evidence")
```

## `EARSConfig`

| Field | Default | Description |
| --- | ---: | --- |
| `motion_candidates` | `256` | Endpoint-inclusive motion samples |
| `flow_scale` | `0.25` | Spatial scale used for Farneback flow |
| `audio_sample_rate` | `22050` | Mono audio decoding rate |
| `audio_segment_seconds` | `1.0` | Audio energy window |
| `audio_weight` | `1.0` | Normalized audio contribution |
| `motion_weight` | `1.0` | Normalized motion contribution |
| `min_gap_seconds` | `0.5` | Temporal suppression radius |

Both evidence weights may be changed independently. Setting a weight to zero
disables that signal in the fused score.

## Errors

EARS raises errors rather than silently returning an invalid selection:

- `FileNotFoundError` for a missing video;
- `RuntimeError` for failed video or frame decoding;
- `subprocess.CalledProcessError` when FFmpeg/FFprobe fails;
- `ValueError` for invalid metadata, configuration, or signal values.

A verified absent audio stream is valid and produces a zero audio signal.
