# Command-Line Interface

Install the package, then run:

```bash
ears-select VIDEO -k K [OPTIONS]
```

Example:

```bash
ears-select data/input/video.mp4 -k 4 \
  --json outputs/video.json \
  --frames-dir outputs/video_frames
```

Options:

| Option | Description |
| --- | --- |
| `-k K` | Required frame count |
| `--json PATH` | Write JSON metadata |
| `--frames-dir PATH` | Decode and save selected JPEG frames |
| `--motion-candidates N` | Motion candidate count |
| `--flow-scale FLOAT` | Farneback spatial scale |
| `--audio-weight FLOAT` | Audio evidence weight |
| `--motion-weight FLOAT` | Motion evidence weight |
| `--min-gap-seconds FLOAT` | Temporal suppression radius |

The JSON document contains no image arrays:

```json
{
  "video_path": "video.mp4",
  "frame_indices": [18, 74, 121, 203],
  "timestamps": [0.6, 2.47, 4.03, 6.77],
  "scores": [0.82, 0.79, 0.91, 0.76],
  "fps": 30.0,
  "num_frames": 300,
  "saved_frames": [
    "outputs/frames/frame_00000018.jpg",
    "outputs/frames/frame_00000074.jpg",
    "outputs/frames/frame_00000121.jpg",
    "outputs/frames/frame_00000203.jpg"
  ]
}
```

`saved_frames` is included only when `--frames-dir` is supplied.
