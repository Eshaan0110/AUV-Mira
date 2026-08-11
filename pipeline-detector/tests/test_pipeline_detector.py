import numpy as np
import cv2
import pytest

from pipeline_detector import (
    ArUcoDetector,
    detect_yellow_pipeline,
    enhance_underwater,
    process_frame,
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

    def test_raises_on_missing_video(self, tmp_path):
        with pytest.raises(IOError):
            run_headless(str(tmp_path / "does_not_exist.mp4"))
