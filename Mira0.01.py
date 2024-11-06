import os
import cv2
import math
import numpy as np
import glob

# Load camera calibration data
data = np.load("./MultiMatrix.npz")
camMatrix = data['camMatrix']
distCoef = data['distCoef']

# Conversion factors for meters to pixels and vice versa
FOCAL_LENGTH_PIXELS = camMatrix[0, 0]
IMAGE_WIDTH = camMatrix[0, 2] * 2
FOCAL_LENGTH_METERS = FOCAL_LENGTH_PIXELS / IMAGE_WIDTH

# Smoothing parameters
pose_history = []
POSE_HISTORY_SIZE = 5  # Number of frames to average

# Function to convert rotation matrix to Euler angles
def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Roll
        y = math.atan2(-rotation_matrix[2, 0], sy)                     # Pitch
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Yaw
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

def estimate_pose(points):
    if len(points) != 4:
        print("4 points required")
        return None

    # Define rectangle dimensions in meters
    rectangle_length_m = 1.4  # Example length in meters
    rectangle_breadth_m = 1.0  # Example breadth in meters

    # Convert rectangle dimensions from meters to pixels
    rectangle_length_px = rectangle_length_m * (FOCAL_LENGTH_PIXELS / (FOCAL_LENGTH_METERS * IMAGE_WIDTH))
    rectangle_breadth_px = rectangle_breadth_m * (FOCAL_LENGTH_PIXELS / (FOCAL_LENGTH_METERS * IMAGE_WIDTH))

    model_points = np.array([
        [0.0, 0.0, 0.0],                     # Point 1 (bottom-left corner)
        [rectangle_length_px, 0.0, 0.0],     # Point 2 (bottom-right corner)
        [rectangle_length_px, rectangle_breadth_px, 0.0], # Point 3 (top-right corner)
        [0.0, rectangle_breadth_px, 0.0]     # Point 4 (top-left corner)
    ], dtype=np.float32)

    image_points = np.array(points, dtype=np.float32)

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points,
                                                                 camMatrix, distCoef)
    if not success:
        print("Pose estimation failed")
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

    distance_in_pixels = np.linalg.norm(translation_vector)
    distance_in_meters = distance_in_pixels * (FOCAL_LENGTH_METERS / IMAGE_WIDTH)

    # Smooth the pose values using a moving average filter
    global pose_history
    pose_history.append((roll, pitch, yaw))
    
    if len(pose_history) > POSE_HISTORY_SIZE:
        pose_history.pop(0)  # Remove the oldest entry

    smoothed_roll = np.mean([p[0] for p in pose_history])
    smoothed_pitch = np.mean([p[1] for p in pose_history])
    smoothed_yaw = np.mean([p[2] for p in pose_history])

    # print(f"Pose estimation successful!")
    print(f"Smoothed Roll: {smoothed_roll:.2f} degrees")
    # print(f"Smoothed Pitch: {smoothed_pitch:.2f} degrees")
    #print(f"Smoothed Yaw: {smoothed_yaw:.2f} degrees")
    # print(f"Distance: {distance_in_meters:.2f} meters")

def detect_gate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 40, 50)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = -1
    largest_corners = None
    
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:   # Check for a quadrilateral
            area = cv2.contourArea(approx)
            if area > largest_area:   # Find the largest rectangle by area
                largest_area = area
                largest_corners = approx.reshape(4,-1) # Reshape to get corners

    return largest_corners

def main():
    image_folder = r'D:\Opencvmain\data2 applied'
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))

    if not image_paths:
        print(f"No images found in folder: {image_folder}")
        exit()

    for image_path in image_paths:
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Error reading image: {image_path}")
            continue

        largest_corners = detect_gate(original_image)

        if largest_corners is not None:
            # Draw corners on the image
            for point in largest_corners:
                cv2.circle(original_image, tuple(point), radius=5, color=(255, 255, 255), thickness=-1)

            # Estimate pose using the detected corners
            estimate_pose(largest_corners)

            # Draw lines connecting the corners for visualization
            cv2.polylines(original_image, [largest_corners], isClosed=True,
                          color=(255, 255, 255), thickness=3)

        cv2.imshow('Detected Gate', original_image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()