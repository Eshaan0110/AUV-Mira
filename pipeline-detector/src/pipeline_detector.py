import cv2
import numpy as np
import argparse
import json
import socket
import sys


class ArUcoDetector:
    """Detects and tracks ArUco markers on pipeline"""
    
    def __init__(self):
        # Use DICT_ARUCO_ORIGINAL as per TAC 2026 rules
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Optimize for underwater detection
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 4.0
        
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Track detected markers
        self.detected_markers = []  # List in order detected
        self.marker_positions = {}  # {marker_id: (x, y)}
        self.confirmation_buffer = {}  # Require multiple detections
        
    def detect(self, frame):
        """
        Detect ArUco markers in frame
        
        Returns:
            list of newly confirmed marker IDs
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for underwater
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Detect markers (new API)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        new_detections = []
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Get marker center
                corner = corners[i][0]
                center_x = np.mean(corner[:, 0])
                center_y = np.mean(corner[:, 1])
                
                # Check if this is a new marker (not seen or far from last position)
                if self._is_new_marker(marker_id, center_x, center_y):
                    # Add to confirmation buffer
                    if marker_id not in self.confirmation_buffer:
                        self.confirmation_buffer[marker_id] = 0
                    
                    self.confirmation_buffer[marker_id] += 1
                    
                    # Confirm after 3 detections
                    if self.confirmation_buffer[marker_id] >= 3:
                        if marker_id not in self.detected_markers:
                            self.detected_markers.append(marker_id)
                            self.marker_positions[marker_id] = (center_x, center_y)
                            new_detections.append(marker_id)
                            print(f"✓ NEW MARKER CONFIRMED: ID {marker_id}")
                        
                        # Clear buffer
                        del self.confirmation_buffer[marker_id]
        
        return new_detections
    
    def _is_new_marker(self, marker_id, x, y, min_distance=200):
        """
        Check if marker is truly new (prevent duplicates)
        Min 0.2m spacing ≈ 200 pixels at typical distance
        """
        if marker_id not in self.marker_positions:
            return True
        
        prev_x, prev_y = self.marker_positions[marker_id]
        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        
        return distance > min_distance
    
    def visualize(self, frame):
        """Draw detected markers on frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Draw all currently visible markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        return frame
    
    def get_marker_list(self):
        """Get ordered list of detected markers"""
        return self.detected_markers


def enhance_underwater(bgr, use_clahe=True, use_specular=True,
                       use_bilateral=True, spec_v=235, spec_s=40,
                       clahe_clip=2.0, clahe_tile=8,
                       bilateral_d=9, bilateral_sigma=75):
    """
    Improve image quality for underwater yellow-detection.
    CLAHE works in LAB space to avoid colour-shift artefacts; specular
    removal inpaints bright halos that confuse the HSV thresholder;
    bilateral filter smooths sensor noise while keeping edges sharp.
    """
    out = bgr

    if use_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=clahe_clip,
                            tileGridSize=(clahe_tile, clahe_tile)).apply(L)
        out = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)

    if use_specular:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)
        spec = ((V > spec_v) & (S < spec_s)).astype(np.uint8) * 255
        spec = cv2.dilate(spec, np.ones((5, 5), np.uint8))
        if np.count_nonzero(spec) > 0:
            out = cv2.inpaint(out, spec, 3, cv2.INPAINT_TELEA)

    if use_bilateral:
        out = cv2.bilateralFilter(out, d=bilateral_d,
                                  sigmaColor=bilateral_sigma,
                                  sigmaSpace=bilateral_sigma)

    return out


def detect_yellow_pipeline(frame, hsv_lower, hsv_upper,
                           preprocess=True,
                           use_clahe=True, use_specular=True,
                           use_bilateral=True,
                           spec_v=235, spec_s=40,
                           close_k=21, **enhance_extra):
    """
    Detect yellow pipeline in frame.

    Returns:
        - detected: bool
        - centroid: (x, y) pixel position
        - normalized_x: -1 (left) to +1 (right), 0 = centered
        - normalized_y: -1 (top) to +1 (bottom), 0 = centered
        - angle: deviation from vertical in degrees (0 = straight ahead)
        - mask: binary mask for visualization
    """
    height, width = frame.shape[:2]

    if preprocess:
        proc = enhance_underwater(frame,
                                  use_clahe=use_clahe,
                                  use_specular=use_specular,
                                  use_bilateral=use_bilateral,
                                  spec_v=spec_v, spec_s=spec_s,
                                  **enhance_extra)
    else:
        proc = frame

    # Convert to HSV
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

    # Create yellow mask
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Elliptical closing kernel bridges pipeline gaps better than a square
    close_k = max(3, close_k | 1)  # ensure odd, >=3
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, None, None, mask

    # Get largest contour (pipeline)
    largest = max(contours, key=cv2.contourArea)

    # Filter small detections (noise)
    if cv2.contourArea(largest) < 1000:
        return None, None, None, None, None, mask

    # Calculate centroid
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, None, None, None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Normalize to [-1, 1] where 0 = center
    norm_x = (cx - width / 2) / (width / 2)
    norm_y = (cy - height / 2) / (height / 2)

    # Angle as deviation from vertical: 0° = perfectly straight,
    # positive = tilted right, negative = tilted left
    [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
    raw = np.arctan2(float(vy[0]), float(vx[0])) * 180 / np.pi
    angle = 90 - abs(raw)
    if raw < 0:
        angle = -angle

    return True, (cx, cy), norm_x, norm_y, angle, mask


def draw_visualization(frame, detected, centroid, norm_x, norm_y, angle):
    """Draw all visualization overlays"""
    vis = frame.copy()
    height, width = frame.shape[:2]
    
    
    if not detected:
        # No pipeline detected
        cv2.putText(vis, "NO PIPELINE DETECTED", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return vis
    
    cx, cy = centroid
    
    # Draw pipeline centroid (big red dot)
    cv2.circle(vis, (cx, cy), 15, (0, 0, 255), -1)  # Filled red
    cv2.circle(vis, (cx, cy), 20, (255, 255, 255), 3)  # White outline
    
    # Draw orientation line
    length = 100
    angle_rad = angle * np.pi / 180
    end_x = int(cx + length * np.cos(angle_rad))
    end_y = int(cy + length * np.sin(angle_rad))
    cv2.line(vis, (cx, cy), (end_x, end_y), (255, 0, 255), 4)  # Purple line
    
    # Draw offset arrow (from center to centroid)
    cv2.arrowedLine(vis, (width//2, height//2), (cx, cy), (0, 255, 0), 3)
    
    # Calculate offset distance in pixels
    offset_pixels = np.sqrt((cx - width//2)**2 + (cy - height//2)**2)
    
    # Display information
    y_pos = 40
    line_height = 35
    
    # Box background for text
    cv2.rectangle(vis, (5, 5), (400, 220), (0, 0, 0), -1)
    cv2.rectangle(vis, (5, 5), (400, 220), (255, 255, 255), 2)
    
    cv2.putText(vis, "PIPELINE DETECTED", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_pos += line_height

    cv2.putText(vis, f"Centroid: ({cx}, {cy})",
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += line_height

    color = (0, 255, 0) if abs(norm_x) < 0.1 else (0, 165, 255)
    cv2.putText(vis, f"Normalized X: {norm_x:+.3f}", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_pos += line_height
    
    cv2.putText(vis, f"Normalized Y: {norm_y:+.3f}", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += line_height
    
    # Offset
    cv2.putText(vis, f"Offset: {offset_pixels:.1f} px", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += line_height
    
    # Angle
    cv2.putText(vis, f"Angle: {angle:.1f} deg", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += line_height
    
    # Status
    if abs(norm_x) < 0.1 and abs(norm_y) < 0.1:
        status = "CENTERED!"
        status_color = (0, 255, 0)
    elif abs(norm_x) < 0.3:
        status = "NEARLY CENTERED"
        status_color = (0, 255, 255)
    else:
        status = "OFF CENTER"
        status_color = (0, 165, 255)
    
    cv2.putText(vis, status, (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Direction hint (what AUV should do)
    y_pos += 50
    if abs(norm_x) > 0.05:
        direction = "MOVE RIGHT" if norm_x < 0 else "MOVE LEFT"
        cv2.putText(vis, f"-> {direction}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return vis


def process_frame(frame, aruco_detector, hsv_lower, hsv_upper,
                  use_clahe=True, use_specular=True, use_bilateral=True,
                  spec_v=235, spec_s=40, close_k=21, **enhance_extra):
    """
    Run the full per-frame pipeline (preprocess -> detect pipeline ->
    detect markers -> visualize) with no GUI calls, so it can be driven
    headlessly (tests, batch processing) as well as from the interactive UI.

    Returns a dict: proc, mask, vis_frame, detected, centroid, norm_x,
    norm_y, angle, new_markers, marker_list.
    """
    proc = enhance_underwater(frame,
                              use_clahe=use_clahe,
                              use_specular=use_specular,
                              use_bilateral=use_bilateral,
                              spec_v=spec_v, spec_s=spec_s,
                              **enhance_extra)

    detected, centroid, norm_x, norm_y, angle, mask = detect_yellow_pipeline(
        proc, hsv_lower, hsv_upper,
        preprocess=False,
        close_k=close_k,
    )

    # ArUco handles its own CLAHE on greyscale, so it gets the raw frame
    new_markers = aruco_detector.detect(frame)

    vis_frame = draw_visualization(frame, detected, centroid, norm_x, norm_y, angle)
    vis_frame = aruco_detector.visualize(vis_frame)

    return {
        "proc": proc,
        "mask": mask,
        "vis_frame": vis_frame,
        "detected": detected,
        "centroid": centroid,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "angle": angle,
        "new_markers": new_markers,
        "marker_list": aruco_detector.get_marker_list(),
    }


def parse_udp_target(text):
    """Parse 'HOST:PORT' into (host, port). Raises ValueError if malformed."""
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise ValueError(f"UDP target {text!r} must look like HOST:PORT")
    try:
        port = int(port)
    except ValueError:
        raise ValueError(f"UDP port in {text!r} is not an integer")
    if not 1 <= port <= 65535:
        raise ValueError(f"UDP port {port} out of range 1..65535")
    return host, port


def make_offset_message(frame_idx, result):
    """Serialise one frame's servoing signal as a compact JSON datagram."""
    msg = {"frame": frame_idx, "detected": bool(result["detected"])}
    if result["detected"]:
        msg["norm_x"] = round(float(result["norm_x"]), 4)
        msg["norm_y"] = round(float(result["norm_y"]), 4)
        msg["angle"] = round(float(result["angle"]), 2)
    return json.dumps(msg, separators=(",", ":")).encode()


DEFAULT_HSV_LOWER = np.array([25, 150, 215])  # [H, S, V]
DEFAULT_HSV_UPPER = np.array([50, 255, 255])


def run_headless(video_path, hsv_lower=None, hsv_upper=None,
                 output_video_path=None, results_path="marker_results.txt",
                 telemetry_path=None, udp_target=None, **enhance_kwargs):
    """
    Run the pipeline over a video with no GUI calls, for onboard/CI use
    where no display is attached. Optionally writes an annotated copy of
    the video, a marker-sequence results file, and a per-frame telemetry
    CSV (frame, detected, norm_x, norm_y, angle) for offline analysis of
    the servoing signal. If udp_target is a (host, port) tuple, each frame's
    offset is also sent as a JSON datagram so a controller can servo on it.

    Extra keyword arguments (use_clahe, use_specular, use_bilateral, spec_v,
    spec_s, close_k, clahe_clip, clahe_tile, bilateral_d, bilateral_sigma) are forwarded to process_frame.

    Returns a stats dict: frame_count, detection_count, detection_rate,
    marker_list.
    """
    if hsv_lower is None:
        hsv_lower = DEFAULT_HSV_LOWER
    if hsv_upper is None:
        hsv_upper = DEFAULT_HSV_UPPER

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    aruco_detector = ArUcoDetector()
    writer = None
    telemetry = None
    sock = None
    if udp_target:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if telemetry_path:
        telemetry = open(telemetry_path, "w", newline="")
        telemetry.write("frame,detected,norm_x,norm_y,angle\n")
    frame_count = 0
    detection_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            result = process_frame(frame, aruco_detector, hsv_lower, hsv_upper,
                                   **enhance_kwargs)
            if result["detected"]:
                detection_count += 1

            if telemetry is not None:
                if result["detected"]:
                    telemetry.write(
                        f"{frame_count},1,{result['norm_x']:.4f},"
                        f"{result['norm_y']:.4f},{result['angle']:.2f}\n")
                else:
                    telemetry.write(f"{frame_count},0,,,\n")

            if sock is not None:
                sock.sendto(make_offset_message(frame_count, result), udp_target)

            if output_video_path:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                writer.write(result["vis_frame"])
    finally:
        cap.release()
        if sock is not None:
            sock.close()
        if telemetry is not None:
            telemetry.close()
        if writer is not None:
            writer.release()

    marker_list = aruco_detector.get_marker_list()
    detection_rate = (detection_count / frame_count * 100) if frame_count else 0.0

    if marker_list and results_path:
        with open(results_path, "w") as f:
            f.write(",".join(map(str, marker_list)))

    return {
        "frame_count": frame_count,
        "detection_count": detection_count,
        "detection_rate": detection_rate,
        "marker_list": marker_list,
    }


def resolve_hsv_bounds(lower=None, upper=None):
    """
    Turn optional [H, S, V] lists into numpy bounds, falling back to the
    defaults. Raises ValueError for out-of-range values or lower > upper,
    so a typo fails fast instead of silently producing an empty mask.
    """
    lower = DEFAULT_HSV_LOWER.copy() if lower is None else np.array(lower)
    upper = DEFAULT_HSV_UPPER.copy() if upper is None else np.array(upper)
    max_vals = np.array([180, 255, 255])
    for name, bound in (("lower", lower), ("upper", upper)):
        if bound.shape != (3,):
            raise ValueError(f"HSV {name} bound must have 3 values")
        if (bound < 0).any() or (bound > max_vals).any():
            raise ValueError(
                f"HSV {name} bound {bound.tolist()} out of range "
                "(H 0-180, S 0-255, V 0-255)")
    if (lower > upper).any():
        raise ValueError(
            f"HSV lower {lower.tolist()} exceeds upper {upper.tolist()}")
    return lower, upper


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="pipeline_detector.py",
        description="Yellow pipeline + ArUco marker detection over a video.")
    parser.add_argument("video", help="path to input video")
    parser.add_argument("--headless", nargs="?", const=True, default=False,
                        metavar="OUTPUT_VIDEO",
                        help="run without a display; optionally write an "
                             "annotated video to OUTPUT_VIDEO")
    parser.add_argument("--telemetry", metavar="CSV",
                        help="(headless) write per-frame offset/angle to CSV")
    parser.add_argument("--udp", metavar="HOST:PORT",
                        help="(headless) send each frame's offset/angle as a "
                             "JSON UDP datagram to HOST:PORT")
    parser.add_argument("--hsv-lower", nargs=3, type=int,
                        metavar=("H", "S", "V"),
                        help="lower HSV bound (default 25 150 215); also the "
                             "starting point of the interactive trackbars")
    parser.add_argument("--hsv-upper", nargs=3, type=int,
                        metavar=("H", "S", "V"),
                        help="upper HSV bound (default 50 255 255)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="disable CLAHE contrast enhancement")
    parser.add_argument("--no-specular", action="store_true",
                        help="disable specular-highlight removal")
    parser.add_argument("--no-bilateral", action="store_true",
                        help="disable bilateral noise filtering")
    parser.add_argument("--spec-v", type=int, default=235, metavar="V",
                        help="specular V threshold (default 235)")
    parser.add_argument("--spec-s", type=int, default=40, metavar="S",
                        help="specular S threshold (default 40)")
    parser.add_argument("--close-k", type=int, default=21, metavar="K",
                        help="morphological close kernel size (default 21)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, metavar="C",
                        help="CLAHE clip limit (default 2.0)")
    parser.add_argument("--clahe-tile", type=int, default=8, metavar="N",
                        help="CLAHE tile grid is N x N (default 8)")
    parser.add_argument("--bilateral-d", type=int, default=9, metavar="D",
                        help="bilateral filter diameter (default 9)")
    parser.add_argument("--bilateral-sigma", type=float, default=75,
                        metavar="S",
                        help="bilateral sigmaColor/sigmaSpace (default 75)")
    parser.add_argument("--results", default="marker_results.txt",
                        metavar="PATH",
                        help="(headless) where to write the marker sequence")
    args = parser.parse_args(argv)
    for name, hi in (("spec_v", 255), ("spec_s", 255), ("close_k", 51)):
        if not 0 <= getattr(args, name) <= hi:
            parser.error(f"--{name.replace('_', '-')} must be in 0..{hi}")
    args.udp_target = None
    if args.udp:
        try:
            args.udp_target = parse_udp_target(args.udp)
        except ValueError as e:
            parser.error(str(e))
    if args.clahe_clip <= 0:
        parser.error("--clahe-clip must be > 0")
    if not 1 <= args.clahe_tile <= 64:
        parser.error("--clahe-tile must be in 1..64")
    if not 1 <= args.bilateral_d <= 25:
        parser.error("--bilateral-d must be in 1..25")
    if args.bilateral_sigma <= 0:
        parser.error("--bilateral-sigma must be > 0")
    return args


def enhance_kwargs_from_args(args):
    """Map parsed CLI flags to process_frame's enhancement keyword args."""
    return dict(use_clahe=not args.no_clahe,
                use_specular=not args.no_specular,
                use_bilateral=not args.no_bilateral,
                spec_v=args.spec_v, spec_s=args.spec_s,
                close_k=args.close_k,
                clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
                bilateral_d=args.bilateral_d,
                bilateral_sigma=args.bilateral_sigma)


TUNER_WINDOW = 'HSV Tuner'


def read_tuner_state(get_pos, window=TUNER_WINDOW):
    """
    Read every interactive trackbar through get_pos(name, window) and return
    (hsv_lower, hsv_upper, enhance_kwargs). Values are clamped to what the
    OpenCV filters accept (a trackbar can sit at 0, which would make a zero
    sized kernel/tile), and CLAHE clip is stored in tenths on its trackbar.
    """
    def pos(name):
        return int(get_pos(name, window))

    lower = np.array([pos('H Min'), pos('S Min'), pos('V Min')])
    upper = np.array([pos('H Max'), pos('S Max'), pos('V Max')])
    kwargs = dict(
        use_clahe=bool(pos('CLAHE')),
        use_specular=bool(pos('Specular')),
        use_bilateral=bool(pos('Bilateral')),
        spec_v=pos('Spec V>'), spec_s=pos('Spec S<'),
        close_k=max(1, pos('Close k')),
        clahe_clip=max(1, pos('CLAHE clip x10')) / 10.0,
        clahe_tile=max(1, pos('CLAHE tile')),
        bilateral_d=max(1, pos('Bilat d')),
        bilateral_sigma=max(1, pos('Bilat sigma')),
    )
    return lower, upper, kwargs


def format_tuned_flags(hsv_lower, hsv_upper, enhance_kwargs):
    """
    Render tuned settings as CLI flags, so values found in the interactive
    tuner can be pasted straight into a headless/onboard run.
    """
    k = enhance_kwargs
    parts = ["--hsv-lower", *map(str, (int(v) for v in hsv_lower)),
             "--hsv-upper", *map(str, (int(v) for v in hsv_upper))]
    for flag, key in (("--no-clahe", "use_clahe"),
                      ("--no-specular", "use_specular"),
                      ("--no-bilateral", "use_bilateral")):
        if not k[key]:
            parts.append(flag)
    parts += ["--spec-v", str(k["spec_v"]), "--spec-s", str(k["spec_s"]),
              "--close-k", str(k["close_k"]),
              "--clahe-clip", f"{k['clahe_clip']:g}",
              "--clahe-tile", str(k["clahe_tile"]),
              "--bilateral-d", str(k["bilateral_d"]),
              "--bilateral-sigma", f"{k['bilateral_sigma']:g}"]
    return " ".join(parts)


def interpret_key(key, paused):
    """Map a cv2.waitKey code to a playback action.

    Returns "quit", "toggle_pause", "step" or None. Stepping only applies
    while paused.
    """
    if key == ord('q'):
        return "quit"
    if key == ord('p'):
        return "toggle_pause"
    if key == ord(' ') and paused:
        return "step"
    return None


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    video_path = args.video
    try:
        hsv_lower, hsv_upper = resolve_hsv_bounds(args.hsv_lower, args.hsv_upper)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    if args.headless:
        output_video_path = args.headless if isinstance(args.headless, str) else None
        stats = run_headless(video_path, hsv_lower, hsv_upper,
                             output_video_path=output_video_path,
                             results_path=args.results,
                             telemetry_path=args.telemetry,
                             udp_target=args.udp_target,
                             **enhance_kwargs_from_args(args))
        print(f"\n{'='*50}")
        print("MISSION RESULTS:")
        print(f"{'='*50}")
        print(f"Frames processed: {stats['frame_count']}")
        print(f"Pipeline detections: {stats['detection_count']}")
        print(f"Detection rate: {stats['detection_rate']:.1f}%")
        print(f"Detected Markers (in order): {stats['marker_list']}")
        print(f"{'='*50}\n")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*50}")
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Total Frames: {total_frames}")
    print(f"{'='*50}\n")
    
    # HSV thresholds tuned for yellow in underwater lighting

    # Initialize ArUco detector
    aruco_detector = ArUcoDetector()

    # Create windows
    cv2.namedWindow('Pipeline Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Preprocessed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)

    # Create trackbars for HSV tuning
    cv2.createTrackbar('H Min', 'HSV Tuner', hsv_lower[0], 180, lambda x: None)
    cv2.createTrackbar('H Max', 'HSV Tuner', hsv_upper[0], 180, lambda x: None)
    cv2.createTrackbar('S Min', 'HSV Tuner', hsv_lower[1], 255, lambda x: None)
    cv2.createTrackbar('S Max', 'HSV Tuner', hsv_upper[1], 255, lambda x: None)
    cv2.createTrackbar('V Min', 'HSV Tuner', hsv_lower[2], 255, lambda x: None)
    cv2.createTrackbar('V Max', 'HSV Tuner', hsv_upper[2], 255, lambda x: None)

    # Preprocessing toggles + tuning
    cv2.createTrackbar('CLAHE',    'HSV Tuner', int(not args.no_clahe),   1,   lambda x: None)
    cv2.createTrackbar('Specular', 'HSV Tuner', int(not args.no_specular),   1,   lambda x: None)
    cv2.createTrackbar('Bilateral','HSV Tuner', int(not args.no_bilateral),   1,   lambda x: None)
    cv2.createTrackbar('Spec V>',   'HSV Tuner', args.spec_v, 255, lambda x: None)
    cv2.createTrackbar('Spec S<',   'HSV Tuner', args.spec_s, 255, lambda x: None)
    cv2.createTrackbar('Close k',   'HSV Tuner', args.close_k, 51,  lambda x: None)
    cv2.createTrackbar('CLAHE clip x10', 'HSV Tuner', int(round(args.clahe_clip * 10)), 100, lambda x: None)
    cv2.createTrackbar('CLAHE tile', 'HSV Tuner', args.clahe_tile, 64, lambda x: None)
    cv2.createTrackbar('Bilat d',   'HSV Tuner', args.bilateral_d, 25, lambda x: None)
    cv2.createTrackbar('Bilat sigma', 'HSV Tuner', int(round(args.bilateral_sigma)), 250, lambda x: None)
    
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  SPACE - Next frame (when paused)")
    print("  Adjust sliders to tune yellow detection\n")
    
    paused = False
    frame_count = 0
    detection_count = 0
    new_frame = False
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break
            frame_count += 1
            new_frame = True
        else:
            ret = True
        
        if ret and frame is not None:
            # Pull current HSV bounds and preprocessing settings from trackbars
            hsv_lower, hsv_upper, enhance_kwargs = read_tuner_state(
                cv2.getTrackbarPos)

            result = process_frame(frame, aruco_detector,
                                   hsv_lower, hsv_upper, **enhance_kwargs)
            proc = result["proc"]
            mask = result["mask"]
            vis_frame = result["vis_frame"]
            detected = result["detected"]

            # Redrawing a paused frame must not inflate the stats
            if detected and new_frame:
                detection_count += 1
            new_frame = False

            # Show detected marker list on frame
            marker_list = result["marker_list"]
            marker_text = f"Detected Markers: {marker_list}" if marker_list else "Detected Markers: None"

            # Draw marker list background
            cv2.rectangle(vis_frame, (5, height - 60), (width - 5, height - 10), (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (5, height - 60), (width - 5, height - 10), (255, 255, 255), 2)

            cv2.putText(vis_frame, marker_text,
                       (10, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Add frame counter
            cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}",
                       (width - 250, height - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if paused:
                cv2.putText(vis_frame, "PAUSED",
                           (width - 150, height - 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show windows
            cv2.imshow('Pipeline Detection', vis_frame)
            cv2.imshow('Mask View', mask)
            cv2.imshow('Preprocessed', proc)

            # Empty window for tuner (just shows trackbars)
            tuner_display = np.zeros((50, 400, 3), dtype=np.uint8)
            cv2.putText(tuner_display, "Adjust HSV values above",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('HSV Tuner', tuner_display)
        
        # Handle keyboard
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        action = interpret_key(key, paused)
        if action == "quit":
            break
        elif action == "toggle_pause":
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif action == "step":
            # Step one frame forward
            ret, nxt = cap.read()
            if ret:
                frame = nxt
                frame_count += 1
                new_frame = True
    
    # Print final statistics
    if frame_count > 0:
        detection_rate = (detection_count / frame_count) * 100
        marker_list = aruco_detector.get_marker_list()
        
        print(f"\n{'='*50}")
        print(f"MISSION RESULTS:")
        print(f"{'='*50}")
        print(f"Pipeline Detection:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Pipeline detections: {detection_count}")
        print(f"  Detection rate: {detection_rate:.1f}%")
        print(f"\nArUco Markers:")
        print(f"  Detected Markers (in order): {marker_list}")
        print(f"  Total unique markers: {len(marker_list)}")
        print(f"\nFinal HSV values:")
        print(f"  Lower: [{hsv_lower[0]}, {hsv_lower[1]}, {hsv_lower[2]}]")
        print(f"  Upper: [{hsv_upper[0]}, {hsv_upper[1]}, {hsv_upper[2]}]")
        print(f"{'='*50}\n")
        
        print("Tuned settings as CLI flags:")
        print("  " + format_tuned_flags(hsv_lower, hsv_upper, enhance_kwargs) + "\n")

        # Save results to file
        if marker_list:
            result_string = ','.join(map(str, marker_list))
            with open('marker_results.txt', 'w') as f:
                f.write(result_string)
            print(f"✓ Results saved to: marker_results.txt")
            print(f"  Marker sequence: {result_string}\n")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()