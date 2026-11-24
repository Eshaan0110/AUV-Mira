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

Optional flags (headless):

- `--telemetry <csv>` - write per-frame `frame,detected,norm_x,norm_y,angle` rows
- `--udp HOST:PORT` - send each frame's `{frame,detected,norm_x,norm_y,angle}` as a JSON UDP datagram (offsets omitted when no pipeline is found) for an external controller
- `--results <path>` - where to write the confirmed marker sequence (default `marker_results.txt`)
- `--hsv-lower H S V` / `--hsv-upper H S V` - override the yellow HSV bounds (defaults `25 150 215` / `50 255 255`); also sets the starting trackbar values in interactive mode
- `--no-clahe` / `--no-specular` / `--no-bilateral` - disable individual preprocessing stages
- `--spec-v V` / `--spec-s S` / `--close-k K` - specular thresholds (default 235/40) and morphological close kernel (default 21); apply to headless runs and seed the interactive trackbars
- `--clahe-clip C` / `--clahe-tile N` - CLAHE clip limit (default 2.0) and N x N tile grid (default 8)
- `--bilateral-d D` / `--bilateral-sigma S` - bilateral filter diameter (default 9) and sigma (default 75)

The interactive tuner also has sliders for CLAHE clip (x10), CLAHE tile, bilateral diameter and sigma. When you quit, it prints the final settings as a ready-to-paste set of CLI flags (`--hsv-lower ... --bilateral-sigma ...`) for headless runs.

## Tests

```bash
uv run pytest tests/ -v
```

Tests exercise `detect_yellow_pipeline`, `enhance_underwater`, and
`ArUcoDetector` against synthetic in-memory frames, so no video file or
display is required.
