import numpy as np
import cv2
import pytest

import pipeline_detector

from pipeline_detector import (
    ArUcoDetector,
    detect_yellow_pipeline,
    enhance_kwargs_from_args,
    enhance_underwater,
    format_tuned_flags,
    interpret_key,
    make_offset_message,
    parse_udp_target,
    parse_args,
    resolve_hsv_bounds,
    process_frame,
    read_tuner_state,
    run_headless,
)

YELLOW_BGR = (0, 255, 255)
HSV_LOWER = np.array([25, 150, 215])
HSV_UPPER = np.array([50, 255, 255])


def blank_frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_vertical_bar(frame, center_x, bar_width=40, margin=90):
    height = frame.shape[0]
    cv2.rectangle(
        frame,
        (center_x - bar_width // 2, margin),
        (center_x + bar_width // 2, height - margin),
        YELLOW_BGR,
        -1,
    )
    return frame


def fake_marker_corners(cx, cy, half_size=25.0):
    return np.array(
        [[
            [cx - half_size, cy - half_size],
            [cx + half_size, cy - half_size],
            [cx + half_size, cy + half_size],
            [cx - half_size, cy + half_size],
        ]],
        dtype=np.float32,
    )


class TestDetectYellowPipeline:
    def test_centered_bar_detected_at_center(self):
        frame = draw_vertical_bar(blank_frame(), center_x=320)
        detected, centroid, norm_x, norm_y, angle, mask = detect_yellow_pipeline(
            frame, HSV_LOWER, HSV_UPPER, preprocess=False
        )

        assert detected is True
        assert centroid[0] == pytest.approx(320, abs=3)
        assert norm_x == pytest.approx(0, abs=0.05)
        assert norm_y == pytest.approx(0, abs=0.05)
        assert angle == pytest.approx(0, abs=2)
        assert mask.shape == frame.shape[:2]

    def test_offset_bar_reports_negative_normalized_x(self):
        frame = draw_vertical_bar(blank_frame(), center_x=160)
        detected, centroid, norm_x, norm_y, angle, mask = detect_yellow_pipeline(
            frame, HSV_LOWER, HSV_UPPER, preprocess=False
        )

        assert detected is True
        assert norm_x < -0.3

    def test_no_yellow_returns_none(self):
        frame = blank_frame()
        detected, centroid, norm_x, norm_y, angle, mask = detect_yellow_pipeline(
            frame, HSV_LOWER, HSV_UPPER, preprocess=False
        )

        assert detected is None
        assert centroid is None
        assert mask.shape == frame.shape[:2]

    def test_small_speck_is_filtered_as_noise(self):
        frame = blank_frame()
        cv2.rectangle(frame, (300, 220), (310, 230), YELLOW_BGR, -1)
        detected, centroid, norm_x, norm_y, angle, mask = detect_yellow_pipeline(
            frame, HSV_LOWER, HSV_UPPER, preprocess=False
        )

        assert detected is None

    def test_tilt_direction_is_opposite_for_mirrored_bars(self):
        right_lean = blank_frame()
        cv2.line(right_lean, (280, 450), (360, 30), YELLOW_BGR, thickness=30)
        left_lean = blank_frame()
        cv2.line(left_lean, (360, 450), (280, 30), YELLOW_BGR, thickness=30)

        _, _, _, _, angle_right, _ = detect_yellow_pipeline(
            right_lean, HSV_LOWER, HSV_UPPER, preprocess=False
        )
        _, _, _, _, angle_left, _ = detect_yellow_pipeline(
            left_lean, HSV_LOWER, HSV_UPPER, preprocess=False
        )

        assert abs(angle_right) > 3
        assert angle_right == pytest.approx(-angle_left, abs=2)

    def test_preprocess_true_does_not_crash_and_still_detects(self):
        frame = draw_vertical_bar(blank_frame(), center_x=320)
        detected, centroid, norm_x, norm_y, angle, mask = detect_yellow_pipeline(
            frame, HSV_LOWER, HSV_UPPER, preprocess=True
        )

        assert detected is True


class TestEnhanceUnderwater:
    def test_output_shape_and_dtype_preserved(self):
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        out = enhance_underwater(frame)

        assert out.shape == frame.shape
        assert out.dtype == frame.dtype

    def test_all_stages_disabled_returns_input_unchanged(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = enhance_underwater(
            frame, use_clahe=False, use_specular=False, use_bilateral=False
        )

        assert np.array_equal(out, frame)


class FakeArucoBackend:
    def __init__(self, corners, ids):
        self._corners = corners
        self._ids = ids

    def detectMarkers(self, gray):
        return self._corners, self._ids, None


class TestArUcoDetector:
    def _feed(self, detector, cx, cy, marker_id=5):
        corners = [fake_marker_corners(cx, cy)]
        ids = np.array([[marker_id]], dtype=np.int32)
        detector.detector = FakeArucoBackend(corners, ids)
        return detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_marker_confirmed_after_three_detections(self):
        detector = ArUcoDetector()

        assert self._feed(detector, 100, 100) == []
        assert self._feed(detector, 100, 100) == []
        assert self._feed(detector, 100, 100) == [5]
        assert detector.get_marker_list() == [5]

    def test_confirmed_marker_is_not_reconfirmed_at_same_spot(self):
        detector = ArUcoDetector()
        for _ in range(3):
            self._feed(detector, 100, 100)

        assert self._feed(detector, 101, 101) == []
        assert detector.get_marker_list() == [5]

    def test_is_new_marker_uses_min_distance_threshold(self):
        detector = ArUcoDetector()
        detector.marker_positions[5] = (100.0, 100.0)

        assert bool(detector._is_new_marker(5, 150.0, 100.0)) is False
        assert bool(detector._is_new_marker(5, 400.0, 100.0)) is True
        assert bool(detector._is_new_marker(7, 100.0, 100.0)) is True


class TestProcessFrame:
    def test_runs_full_pipeline_headlessly_and_reports_detection(self):
        frame = draw_vertical_bar(blank_frame(), center_x=320)
        detector = ArUcoDetector()

        result = process_frame(frame, detector, HSV_LOWER, HSV_UPPER)

        assert result["detected"] is True
        assert result["centroid"][0] == pytest.approx(320, abs=3)
        assert result["mask"].shape == frame.shape[:2]
        assert result["proc"].shape == frame.shape
        assert result["vis_frame"].shape == frame.shape
        assert result["new_markers"] == []
        assert result["marker_list"] == []

    def test_no_pipeline_still_returns_visualized_frame(self):
        frame = blank_frame()
        detector = ArUcoDetector()

        result = process_frame(frame, detector, HSV_LOWER, HSV_UPPER)

        assert result["detected"] is None
        assert result["centroid"] is None
        assert result["vis_frame"].shape == frame.shape

    def test_confirmed_marker_surfaces_in_new_markers_and_list(self):
        frame = blank_frame()
        detector = ArUcoDetector()
        corners = [fake_marker_corners(100, 100)]
        ids = np.array([[5]], dtype=np.int32)
        detector.detector = FakeArucoBackend(corners, ids)

        for _ in range(2):
            result = process_frame(frame, detector, HSV_LOWER, HSV_UPPER)
            assert result["new_markers"] == []

        result = process_frame(frame, detector, HSV_LOWER, HSV_UPPER)
        assert result["new_markers"] == [5]
        assert result["marker_list"] == [5]


def write_synthetic_video(path, num_frames=5, width=320, height=240, center_x=160):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for _ in range(num_frames):
        frame = draw_vertical_bar(
            blank_frame(width, height), center_x=center_x, margin=40
        )
        writer.write(frame)
    writer.release()


class TestRunHeadless:
    def test_reports_stats_with_no_output_video(self, tmp_path):
        video_path = tmp_path / "in.mp4"
        write_synthetic_video(video_path)

        stats = run_headless(
            str(video_path),
            hsv_lower=HSV_LOWER,
            hsv_upper=HSV_UPPER,
            results_path=str(tmp_path / "markers.txt"),
        )

        assert stats["frame_count"] == 5
        assert stats["detection_count"] == 5
        assert stats["detection_rate"] == pytest.approx(100.0)
        assert stats["marker_list"] == []

    def test_writes_annotated_output_video(self, tmp_path):
        video_path = tmp_path / "in.mp4"
        out_path = tmp_path / "out.mp4"
        write_synthetic_video(video_path)

        run_headless(
            str(video_path),
            hsv_lower=HSV_LOWER,
            hsv_upper=HSV_UPPER,
            output_video_path=str(out_path),
            results_path=str(tmp_path / "markers.txt"),
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_writes_per_frame_telemetry_csv(self, tmp_path):
        video_path = tmp_path / "in.mp4"
        csv_path = tmp_path / "telemetry.csv"
        write_synthetic_video(video_path)

        run_headless(
            str(video_path),
            hsv_lower=HSV_LOWER,
            hsv_upper=HSV_UPPER,
            results_path=str(tmp_path / "markers.txt"),
            telemetry_path=str(csv_path),
        )

        lines = csv_path.read_text().strip().splitlines()
        assert lines[0] == "frame,detected,norm_x,norm_y,angle"
        assert len(lines) == 1 + 5
        first = lines[1].split(",")
        assert first[0] == "1" and first[1] == "1"
        float(first[2]); float(first[3]); float(first[4])

    def test_forwards_enhancement_options(self, tmp_path, monkeypatch):
        import pipeline_detector as pd
        video_path = tmp_path / "in.mp4"
        write_synthetic_video(video_path)
        seen = []
        real = pd.process_frame

        def spy(*a, **kw):
            seen.append(kw)
            return real(*a, **kw)

        monkeypatch.setattr(pd, "process_frame", spy)
        run_headless(str(video_path), hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                     results_path=str(tmp_path / "m.txt"),
                     use_clahe=False, use_bilateral=False, close_k=5)
        assert seen and all(kw["use_clahe"] is False and kw["close_k"] == 5
                            for kw in seen)

    def test_raises_on_missing_video(self, tmp_path):
        with pytest.raises(IOError):
            run_headless(str(tmp_path / "does_not_exist.mp4"))


class TestParseArgs:
    def test_interactive_default(self):
        args = parse_args(["clip.mp4"])
        assert args.video == "clip.mp4"
        assert args.headless is False

    def test_headless_without_output(self):
        assert parse_args(["clip.mp4", "--headless"]).headless is True

    def test_headless_with_output_and_telemetry(self):
        args = parse_args(["clip.mp4", "--headless", "out.mp4",
                           "--telemetry", "t.csv"])
        assert args.headless == "out.mp4"
        assert args.telemetry == "t.csv"


class TestEnhancementFlags:
    def test_defaults_match_process_frame(self):
        kw = enhance_kwargs_from_args(parse_args(["c.mp4"]))
        assert kw == dict(use_clahe=True, use_specular=True,
                          use_bilateral=True, spec_v=235, spec_s=40,
                          close_k=21, clahe_clip=2.0, clahe_tile=8,
                          bilateral_d=9, bilateral_sigma=75)

    def test_flags_map_to_kwargs(self):
        args = parse_args(["c.mp4", "--no-clahe", "--no-bilateral",
                           "--spec-v", "250", "--close-k", "9"])
        kw = enhance_kwargs_from_args(args)
        assert kw["use_clahe"] is False and kw["use_bilateral"] is False
        assert kw["use_specular"] is True
        assert kw["spec_v"] == 250 and kw["close_k"] == 9

    def test_out_of_range_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["c.mp4", "--close-k", "99"])


class TestHsvBounds:
    def test_defaults(self):
        lo, hi = resolve_hsv_bounds()
        assert lo.tolist() == [25, 150, 215]
        assert hi.tolist() == [50, 255, 255]

    def test_override(self):
        lo, hi = resolve_hsv_bounds([20, 100, 100], [40, 255, 255])
        assert lo.tolist() == [20, 100, 100]
        assert hi.tolist() == [40, 255, 255]

    def test_defaults_not_aliased(self):
        lo, _ = resolve_hsv_bounds()
        lo[0] = 0
        assert resolve_hsv_bounds()[0][0] == 25

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            resolve_hsv_bounds([0, 0, 0], [181, 255, 255])

    def test_lower_above_upper_rejected(self):
        with pytest.raises(ValueError):
            resolve_hsv_bounds([60, 0, 0], [50, 255, 255])

    def test_cli_flags_parse(self):
        args = parse_args(["c.mp4", "--hsv-lower", "20", "90", "100",
                           "--hsv-upper", "40", "255", "255"])
        assert args.hsv_lower == [20, 90, 100]
        assert args.hsv_upper == [40, 255, 255]


class TestFilterParams:
    def test_flags_map_to_kwargs(self):
        kw = enhance_kwargs_from_args(parse_args(
            ["c.mp4", "--clahe-clip", "4.5", "--clahe-tile", "4",
             "--bilateral-d", "5", "--bilateral-sigma", "40"]))
        assert kw["clahe_clip"] == 4.5 and kw["clahe_tile"] == 4
        assert kw["bilateral_d"] == 5 and kw["bilateral_sigma"] == 40

    @pytest.mark.parametrize("flag,val", [
        ("--clahe-clip", "0"), ("--clahe-tile", "0"),
        ("--clahe-tile", "100"), ("--bilateral-d", "0"),
        ("--bilateral-sigma", "-1")])
    def test_out_of_range_rejected(self, flag, val):
        with pytest.raises(SystemExit):
            parse_args(["c.mp4", flag, val])

    def test_clahe_clip_changes_output(self):
        rng = np.random.default_rng(0)
        img = rng.integers(80, 120, (64, 64, 3), dtype=np.uint8)
        kw = dict(use_specular=False, use_bilateral=False)
        weak = enhance_underwater(img, clahe_clip=1.0, **kw)
        strong = enhance_underwater(img, clahe_clip=8.0, **kw)
        assert not np.array_equal(weak, strong)

    def test_params_forwarded_through_process_frame(self):
        frame = np.zeros((48, 64, 3), np.uint8)
        res = process_frame(frame, ArUcoDetector(), np.array([25, 150, 215]),
                            np.array([50, 255, 255]), clahe_clip=3.0,
                            clahe_tile=4, bilateral_d=5, bilateral_sigma=30)
        assert res["proc"].shape == frame.shape


class TestUdpOutput:
    def test_parse_udp_target(self):
        assert parse_udp_target("127.0.0.1:9000") == ("127.0.0.1", 9000)

    @pytest.mark.parametrize("bad", ["localhost", ":9000", "h:abc", "h:0", "h:70000"])
    def test_parse_udp_target_rejects_bad(self, bad):
        with pytest.raises(ValueError):
            parse_udp_target(bad)

    def test_parse_args_rejects_bad_udp(self):
        with pytest.raises(SystemExit):
            parse_args(["v.mp4", "--udp", "nonsense"])

    def test_parse_args_sets_udp_target(self):
        assert parse_args(["v.mp4", "--udp", "h:5"]).udp_target == ("h", 5)

    def test_message_undetected_has_no_offsets(self):
        import json
        msg = json.loads(make_offset_message(3, {"detected": False}))
        assert msg == {"frame": 3, "detected": False}

    def test_headless_sends_one_datagram_per_frame(self, tmp_path):
        import json, socket
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(2)
        video_path = tmp_path / "in.mp4"
        write_synthetic_video(video_path)
        run_headless(str(video_path), hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                     results_path=str(tmp_path / "m.txt"),
                     udp_target=("127.0.0.1", rx.getsockname()[1]))
        msgs = [json.loads(rx.recv(1024)) for _ in range(5)]
        rx.close()
        assert [m["frame"] for m in msgs] == [1, 2, 3, 4, 5]
        assert msgs[0]["detected"] is True
        assert {"norm_x", "norm_y", "angle"} <= msgs[0].keys()


class TestTuner:
    VALUES = {
        'H Min': 20, 'H Max': 40, 'S Min': 100, 'S Max': 255,
        'V Min': 200, 'V Max': 255, 'CLAHE': 1, 'Specular': 0,
        'Bilateral': 1, 'Spec V>': 230, 'Spec S<': 30, 'Close k': 15,
        'CLAHE clip x10': 35, 'CLAHE tile': 4, 'Bilat d': 7,
        'Bilat sigma': 50,
    }

    def _read(self, **over):
        vals = {**self.VALUES, **over}
        return read_tuner_state(lambda name, win: vals[name])

    def test_reads_bounds_and_settings(self):
        lower, upper, kw = self._read()
        assert lower.tolist() == [20, 100, 200]
        assert upper.tolist() == [40, 255, 255]
        assert kw["use_specular"] is False and kw["use_clahe"] is True
        assert kw["clahe_clip"] == 3.5 and kw["clahe_tile"] == 4

    def test_zero_trackbars_are_clamped(self):
        _, _, kw = self._read(**{'Close k': 0, 'CLAHE clip x10': 0,
                                 'CLAHE tile': 0, 'Bilat d': 0,
                                 'Bilat sigma': 0})
        assert kw["close_k"] == 1 and kw["clahe_tile"] == 1
        assert kw["clahe_clip"] == 0.1 and kw["bilateral_d"] == 1
        assert kw["bilateral_sigma"] == 1

    def test_clamped_state_runs_through_enhancer(self):
        _, _, kw = self._read(**{'CLAHE tile': 0, 'Bilat d': 0})
        frame = np.full((40, 40, 3), 90, np.uint8)
        kw.pop("close_k")
        assert enhance_underwater(frame, **kw).shape == frame.shape

    def test_format_flags_round_trip_through_parse_args(self):
        lower, upper, kw = self._read()
        text = format_tuned_flags(lower, upper, kw)
        args = parse_args(["v.mp4"] + text.split())
        assert args.hsv_lower == [20, 100, 200]
        assert args.hsv_upper == [40, 255, 255]
        assert args.no_specular and not args.no_clahe
        assert enhance_kwargs_from_args(args) == kw


class TestInterpretKey:
    def test_quit_and_pause_keys(self):
        assert interpret_key(ord("q"), False) == "quit"
        assert interpret_key(ord("p"), True) == "toggle_pause"

    def test_space_steps_only_when_paused(self):
        assert interpret_key(ord(" "), True) == "step"
        assert interpret_key(ord(" "), False) is None

    def test_unknown_key_is_ignored(self):
        assert interpret_key(ord("z"), True) is None


class TestInteractiveLoop:
    def _run(self, monkeypatch, tmp_path, keys, capsys):
        video = tmp_path / "in.mp4"
        write_synthetic_video(video, num_frames=5)
        key_iter = iter(keys)
        for name in ("namedWindow", "imshow", "createTrackbar"):
            monkeypatch.setattr(cv2, name, lambda *a, **k: None)
        monkeypatch.setattr(cv2, "getTrackbarPos", lambda name, win: {
            "H Min": 20, "H Max": 40, "S Min": 100, "S Max": 255,
            "V Min": 100, "V Max": 255, "CLAHE": 0, "Specular": 0,
            "Bilateral": 0, "Spec V>": 200, "Spec S<": 60, "Close k": 5,
            "CLAHE clip x10": 20, "CLAHE tile": 8, "Bilat d": 9,
            "Bilat sigma": 75}[name])
        monkeypatch.setattr(cv2, "waitKey", lambda delay: next(key_iter, ord("q")))
        monkeypatch.chdir(tmp_path)
        pipeline_detector.main([str(video), "--no-clahe"])
        return capsys.readouterr().out

    def test_paused_redraws_do_not_inflate_counts(self, monkeypatch, tmp_path, capsys):
        # Pause on frame 1, press unrelated keys (redraws), then quit.
        out = self._run(monkeypatch, tmp_path,
                        [ord("p"), ord("x"), ord("x"), ord("q")], capsys)
        assert "Frames processed: 1" in out
        assert "Pipeline detections: 1" in out
