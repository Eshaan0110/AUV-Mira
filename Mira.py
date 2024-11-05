import os
import cv2
import math
import numpy as np


data = np.load("./MultiMatrix.npz")
camMatrix = data['camMatrix']
distCoef = data['distCoef']

# Extract focal length (assuming fx and fy are equal)
FOCAL_LENGTH_PIXELS = camMatrix[0, 0]
IMAGE_WIDTH = camMatrix[0, 2] * 2  # cx = IMAGE_WIDTH / 2

# Calculate Horizontal Field of View (HFOV)
HFOV = 2 * np.degrees(np.arctan((IMAGE_WIDTH / (2 * FOCAL_LENGTH_PIXELS))))

# Calculate Sensor Width using FOV (in meters)
FOCAL_LENGTH_METERS = FOCAL_LENGTH_PIXELS / IMAGE_WIDTH  # Focal length proportionate to sensor width
SENSOR_WIDTH = 2 * FOCAL_LENGTH_METERS * math.tan(math.radians(HFOV / 2))

# Conversion factor (how many pixels per meter)
meters_to_pixels = FOCAL_LENGTH_PIXELS / SENSOR_WIDTH
pixels_to_meters = 1 / meters_to_pixels  # To convert pixels back to meters


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles (yaw, pitch, roll).
    The angles are returned in degrees.
    """
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    # Check if near singularity to avoid gimbal lock issues
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Roll
        y = math.atan2(-rotation_matrix[2, 0], sy)                   # Pitch
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
    rectangle_breadth_m = 1  # Example breadth in meters

    # Convert rectangle dimensions from meters to pixels
    rectangle_length_px = rectangle_length_m * meters_to_pixels
    rectangle_breadth_px = rectangle_breadth_m * meters_to_pixels

    model_points = np.array([
        [0.0, 0.0, 0.0],  # Point 1 (bottom-left corner)
        [rectangle_length_px, 0.0, 0.0],  # Point 2 (bottom-right corner)
        [rectangle_length_px, rectangle_breadth_px, 0.0],  # Point 3 (top-right corner)
        [0.0, rectangle_breadth_px, 0.0]  # Point 4 (top-left corner)
    ], dtype=np.float32)

    image_points = np.array(points, dtype=np.float32)

    # Centered at Y-axis
    camera_matrix = camMatrix

    # Get from calibration file
    dist_coeffs = distCoef

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        print("Pose estimation failed")
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract roll, pitch, and yaw
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

    # Distance from the camera to the rectangle
    distance_in_pixels = np.linalg.norm(translation_vector)
    distance_in_meters = distance_in_pixels * pixels_to_meters

    if(yaw<-90):
        yaw= -(180+yaw)
    else:
        yaw= (180-yaw)

    print(f"Pose estimation successful!")
    print(f"Roll: {roll:.2f} degrees")
    print(f"Pitch: {pitch:.2f} degrees")
    print(f"Yaw: {yaw:.2f} degrees")
    print(f"Distance: {distance_in_meters:.2f} meters")

    return roll, pitch, yaw, distance_in_meters

def Filters(image):
    image_copy2 = image.copy()

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)


    clahe_for_l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    clahe_for_a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    clahe_for_b = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    clahe_b = clahe_for_b.apply(b_channel)
    clahe_a = clahe_for_a.apply(a_channel)
    clahe_l = clahe_for_l.apply(l_channel)

    lab_clahe = cv2.merge((clahe_l, clahe_a, clahe_b))

    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    balanced_image = image_clahe
    # cv2.32how("Mid",balanced_image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)

    exposure_factor = (-0.0044117) * brightness + 1.695287

    balanced_image = np.clip(balanced_image * exposure_factor, 0, 255).astype(np.uint8)

    scale = 1.2
    delta = 0
    ddepth = cv2.CV_16S
    blurred_image = cv2.GaussianBlur(balanced_image, (3, 3), 0)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


    # cv2.imshow("Segment", grad)

    _, grad = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)

    linesP = cv2.HoughLinesP(grad, 1, np.pi / 180,50, None, 50, 10)

    if linesP is not None:
        extended_lines = []

        for i in range(0, len(linesP)):
            l = linesP[i][0]

            dx = l[2] - l[0]
            dy = l[3] - l[1]

            length = np.sqrt(dx**2 + dy**2)

            direction = (dx / length, dy / length)

            extend_length = 100
            new_x1 = int(l[0] - direction[0] * extend_length)
            new_y1 = int(l[1] - direction[1] * extend_length)
            new_x2 = int(l[2] + direction[0] * extend_length)
            new_y2 = int(l[3] + direction[1] * extend_length)

            ###cv2.line(image_copy2, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 3, cv2.LINE_AA)

            extended_lines.append([new_x1, new_y1, new_x2, new_y2])

        intersections = []
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):
                # Unpack both lines
                x1, y1, x2, y2 = extended_lines[i]
                x3, y3, x4, y4 = extended_lines[j]

                # Line 1
                a1 = y2 - y1
                b1 = x1 - x2
                c1 = a1 * x1 + b1 * y1

                # Line 2
                a2 = y4 - y3
                b2 = x3 - x4
                c2 = a2 * x3 + b2 * y3

                # determinant
                det = a1 * b2 - a2 * b1

                if det != 0:
                    x = (b2 * c1 - b1 * c2) / det
                    y = (a1 * c2 - a2 * c1) / det
                    intersection = (int(x), int(y))

                    intersections.append(intersection)

                    ###cv2.circle(image_copy2, intersection, 5, (0, 255, 0), -1)

        def distance_between(point1, point2):
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


        cluster_threshold = 50

        if intersections:
            clusters = []
            used_points = set()

            for i, point1 in enumerate(intersections):
                if i in used_points:
                    continue
                cluster = [point1]
                used_points.add(i)

                for j, point2 in enumerate(intersections):
                    if j in used_points:
                        continue
                    if distance_between(point1, point2) < cluster_threshold:
                        cluster.append(point2)
                        used_points.add(j)


                Min_Cluster_Size = 80
                if len(cluster) > Min_Cluster_Size:
                    avg_x = int(np.mean([p[0] for p in cluster]))
                    avg_y = int(np.mean([p[1] for p in cluster]))
                    avg_point = (avg_x, avg_y)

                    clusters.append(avg_point)

            for point in clusters:
                cv2.circle(image_copy2, point, 5, (0, 255, 255), -1)

            estimate_pose(clusters)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", image_copy2)
    return image_copy2


# image = cv2.imread("Localisation/Gate/Half_time/frame_005605.png")
# Filters(image)



import os
import glob

image_folder = r'D:\Opencvmain\Half_time'
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
if not image_paths:
    print(f"No images found in folder: {image_folder}")
    exit()

first_image = cv2.imread(image_paths[0])
if first_image is None:
    print("Error reading images. Exiting...")
    exit()

height, width, _ = first_image.shape

cv2.namedWindow('Original and Filtered Footage', cv2.WINDOW_NORMAL)

for image_path in image_paths:
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error reading image: {image_path}")
        continue

    filtered_image = Filters(original_image)
    combined_frame = cv2.hconcat([original_image, filtered_image])

    cv2.imshow('Original and Filtered Footage', combined_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
