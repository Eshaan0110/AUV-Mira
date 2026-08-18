# pipeline-detector

Computer-vision module for AUV-Mira's yellow pipeline following and ArUco
marker tracking (TAC 2026). Given underwater video, it enhances each frame,
thresholds for the yellow pipeline, computes a centroid/offset/heading for
visual servoing, and tracks ArUco markers (`DICT_ARUCO_ORIGINAL`) placed
along the pipeline.

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
cd pipeline-detector
uv sync
```

## Usage

### Interactive (HSV tuning UI)

```bash
uv run python main.py <video_path>
```

Opens windows showing the annotated detection, binary mask, preprocessed
frame, and HSV tuning sliders. Controls:

- `q` - quit
- `p` - pause/resume
- `SPACE` - step one frame forward while paused
- Adjust the HSV/CLAHE/specular/bilateral sliders to tune yellow detection live

On exit, prints detection stats and the confirmed marker sequence, and
writes the marker sequence to `marker_results.txt` if any were found.

### Headless (no display, for onboard/CI use)

```bash
uv run python main.py <video_path> --headless [output_video_path]
```

Runs the same detection pipeline with no OpenCV GUI calls. Prints frame
count, detection rate, and marker sequence to stdout. If `output_video_path`
is given, writes an annotated copy of the video there.

## Tests

```bash
uv run pytest tests/ -v
```

Tests exercise `detect_yellow_pipeline`, `enhance_underwater`, and
`ArUcoDetector` against synthetic in-memory frames, so no video file or
display is required.
