import cv2
import numpy as np
import yaml


def calibrate_camera_auto():
    """
    Automatic camera calibration using live webcam.
    It collects good checkerboard poses and computes:
    - Intrinsic matrix (K)
    - Distortion coefficients
    """

    # --- Checkerboard settings ---
    CHECKERBOARD = (9, 6)   # inner corners (columns, rows)
    REQUIRED_SAMPLES = 20   # number of good frames to collect

    # Criteria for improving corner accuracy
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    # --- Prepare 3D world points (Z = 0 since board is flat) ---
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    objpoints = []  # 3D points in world
    imgpoints = []  # 2D points in image

    cap = cv2.VideoCapture(0)

    print("Move checkerboard slowly in different angles.")
    print("Collecting frames automatically...")

    last_rvec = None
    last_tvec = None

    # --- Collect frames ---
    while len(objpoints) < REQUIRED_SAMPLES:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, None
        )

        display = frame.copy()

        if found:

            # Make corner detection more accurate
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            cv2.drawChessboardCorners(
                display, CHECKERBOARD, corners_refined, found
            )

            # Temporary fake intrinsic matrix
            # Used only to check pose difference
            fake_K = np.array([
                [1000, 0, gray.shape[1] / 2],
                [0, 1000, gray.shape[0] / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                objp, corners_refined, fake_K, None
            )

            if last_rvec is None:
                save_frame = True
            else:
                # Check if camera moved enough
                rot_diff = np.linalg.norm(rvec - last_rvec)
                trans_diff = np.linalg.norm(tvec - last_tvec)

                save_frame = rot_diff > 0.1 or trans_diff > 10

            if save_frame:
                print(f"Captured sample {len(objpoints) + 1}")
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                last_rvec = rvec
                last_tvec = tvec

        cv2.imshow("Auto Calibration", display)

        if cv2.waitKey(1) == 27:  # ESC to stop early
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print("Not enough samples collected.")
        return None, None

    print("Running calibration...")

    # --- Main calibration step ---
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\nIntrinsic Matrix (K):\n", K)
    print("\nDistortion Coefficients:\n", dist)

    # --- Calculate reprojection error ---
    total_error = 0

    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )

        error = cv2.norm(
            imgpoints[i], projected_points, cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / len(objpoints)

    print("\nMean Reprojection Error:", mean_error)

    # --- Save calibration ---
    calibration_data = {
        "K": K.tolist(),
        "distortion": dist.tolist()
    }

    with open("calibration.yaml", "w") as f:
        yaml.dump(calibration_data, f)

    print("Calibration saved to calibration.yaml")

    return K, dist