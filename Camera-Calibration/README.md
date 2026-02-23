# 📷 Camera Calibration (Live Webcam)

A Python script that automatically calibrates your camera using a checkerboard pattern and your webcam — no manual photo collection needed.

---

## What does "calibrating a camera" even mean?

Every camera lens introduces some distortion — straight lines look slightly curved, and pixel positions don't perfectly correspond to real-world angles. Calibration figures out *how much* distortion there is and saves the math to fix it.

Two things get computed:

- **Intrinsic Matrix `K`** — captures focal length and the optical center of the camera
- **Distortion Coefficients `D`** — captures how much the lens bends light (barrel/fisheye effect)

Once you have these, you can undistort any image from that camera, which is essential before running detection algorithms.

---

## How it works

1. You hold a checkerboard in front of your webcam and move it around
2. The script automatically detects the checkerboard corners in each frame
3. It only saves a frame if the board has **moved enough** from the last saved pose (so you don't collect 20 identical frames)
4. Once 20 good frames are collected, it runs calibration and saves the results to `calibration.yaml`

---

## Requirements

```bash
pip install opencv-python numpy pyyaml
```

You also need a printed checkerboard with **9×6 inner corners** (easy to find/print online).

---

## Usage

```python
from calibrate import calibrate_camera_auto

K, dist = calibrate_camera_auto()
```

Move the checkerboard slowly in different angles and distances in front of the webcam. The script will print progress as it collects samples. Press `ESC` to stop early.

---

## Output

A `calibration.yaml` file containing:

```yaml
K:
  - [fx, 0, cx]
  - [0, fy, cy]
  - [0, 0, 1]
distortion:
  - [k1, k2, p1, p2, k3]
```

The **mean reprojection error** is also printed — this tells you how accurate the calibration is. Anything under `1.0 px` is good.

---

## What I learned

- Using a fake/approximate `K` matrix with `solvePnP` is a valid trick to check relative pose change between frames — you don't need a perfect calibration just to decide "did the board move enough?"
- Collecting diverse poses (tilted, rotated, at different distances, near the corners of the frame) matters more than just collecting many frames
- `calibration.yaml` can be loaded in any downstream script and used directly — you only need to calibrate once per camera setup