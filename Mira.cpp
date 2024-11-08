#include <iostream>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

using namespace std;
using namespace cv;

Mat camMatrix, distCoef;
double FOCAL_LENGTH_PIXELS, IMAGE_WIDTH, HFOV, FOCAL_LENGTH_METERS, SENSOR_WIDTH, meters_to_pixels, pixels_to_meters;

Vec3d rotation_matrix_to_euler_angles(Mat rotation_matrix) {
    double sy = sqrt(rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(0, 0) +
                    rotation_matrix.at<double>(1, 0) * rotation_matrix.at<double>(1, 0));

    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2));
        y = atan2(-rotation_matrix.at<double>(2, 0), sy);
        z = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0));
    } else {
        x = atan2(-rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(1, 1));
        y = atan2(-rotation_matrix.at<double>(2, 0), sy);
        z = 0;
    }

    return Vec3d(x * 180 / CV_PI, y * 180 / CV_PI, z * 180 / CV_PI);
}

vector<Point2f> estimate_pose(vector<Point2f> points) {
    if (points.size() != 4) {
        cout << "4 points required" << endl;
        return {};
    }

    double rectangle_length_m = 1.4;
    double rectangle_breadth_m = 1.0;

    double rectangle_length_px = rectangle_length_m * meters_to_pixels;
    double rectangle_breadth_px = rectangle_breadth_m * meters_to_pixels;

    vector<Point3f> model_points = {
        Point3f(0.0, 0.0, 0.0),
        Point3f(rectangle_length_px, 0.0, 0.0),
        Point3f(rectangle_length_px, rectangle_breadth_px, 0.0),
        Point3f(0.0, rectangle_breadth_px, 0.0)
    };

    vector<Point2f> image_points(points);

    Mat rotation_vector, translation_vector;
    bool success = solvePnP(model_points, image_points, camMatrix, distCoef, rotation_vector, translation_vector);

    if (!success) {
        cout << "Pose estimation failed" << endl;
        return {};
    }

    Mat rotation_matrix;
    Rodrigues(rotation_vector, rotation_matrix);

    Vec3d euler_angles = rotation_matrix_to_euler_angles(rotation_matrix);
    double roll = euler_angles[0], pitch = euler_angles[1], yaw = euler_angles[2];

    double distance_in_pixels = norm(translation_vector);
    double distance_in_meters = distance_in_pixels * pixels_to_meters;

    if (yaw < -90) {
        yaw = -(180 + yaw);
    } else {
        yaw = (180 - yaw);
    }

    cout << "Roll: " << roll << " degrees" << endl;
    cout << "Pitch: " << pitch << " degrees" << endl;
    cout << "Yaw: " << yaw << " degrees" << endl;
    cout << "Distance: " << distance_in_meters << " meters" << endl;

    return image_points;
}

Mat Filters(Mat image) {
    Mat image_copy2 = image.clone();

    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    vector<Mat> lab_channels;
    split(lab_image, lab_channels);
    Mat l_channel = lab_channels[0], a_channel = lab_channels[1], b_channel = lab_channels[2];

    Ptr<CLAHE> clahe_for_l = createCLAHE(4.0, Size(16, 16));
    Ptr<CLAHE> clahe_for_a = createCLAHE(2.0, Size(4, 4));
    Ptr<CLAHE> clahe_for_b = createCLAHE(2.0, Size(4, 4));

    Mat clahe_b;
    clahe_for_b->apply(b_channel, clahe_b);
    Mat clahe_a;
    clahe_for_a->apply(a_channel, clahe_a);
    Mat clahe_l;
    clahe_for_l->apply(l_channel, clahe_l);

    vector<Mat> lab_clahe_channels = {clahe_l, clahe_a, clahe_b};
    Mat lab_clahe;
    merge(lab_clahe_channels, lab_clahe);

    Mat image_clahe;
    cvtColor(lab_clahe, image_clahe, COLOR_Lab2BGR);

    Mat balanced_image = image_clahe;

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    double brightness = mean(gray_image)[0];

    double exposure_factor = (-0.0044117) * brightness + 1.695287;

    convertScaleAbs(balanced_image * exposure_factor, balanced_image, 1, 0);


    Mat blurred_image;
    GaussianBlur(balanced_image, blurred_image, Size(3, 3), 0);
    cvtColor(blurred_image, gray_image, COLOR_BGR2GRAY);

    Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    Sobel(gray_image, grad_x, CV_16S, 1, 0, 3, 1.2, 0, BORDER_DEFAULT);
    Sobel(gray_image, grad_y, CV_16S, 0, 1, 3, 1.2, 0, BORDER_DEFAULT);
    
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    threshold(grad, grad, 50, 255, THRESH_BINARY);

    vector<Vec4i> linesP;
    HoughLinesP(grad, linesP, 1, CV_PI / 180, 50, 50, 10);

    if (!linesP.empty()) {
        vector<Vec4i> extended_lines;

        for (const auto& l : linesP) {
            double dx = l[2] - l[0];
            double dy = l[3] - l[1];
            double length = sqrt(dx * dx + dy * dy);
            double direction_x = dx / length;
            double direction_y = dy / length;

            double extend_length = 100;
            int new_x1 = static_cast<int>(l[0] - direction_x * extend_length);
            int new_y1 = static_cast<int>(l[1] - direction_y * extend_length);
            int new_x2 = static_cast<int>(l[2] + direction_x * extend_length);
            int new_y2 = static_cast<int>(l[3] + direction_y * extend_length);

            extended_lines.emplace_back(new_x1, new_y1, new_x2, new_y2);
            // cv2.line(image_copy2, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 3, cv2.LINE_AA)
        }

        vector<Point2f> intersections;
        for (size_t i = 0; i < extended_lines.size(); i++) {
            for (size_t j = i + 1; j < extended_lines.size(); j++) {
                int x1 = extended_lines[i][0], y1 = extended_lines[i][1];
                int x2 = extended_lines[i][2], y2 = extended_lines[i][3];
                int x3 = extended_lines[j][0], y3 = extended_lines[j][1];
                int x4 = extended_lines[j][2], y4 = extended_lines[j][3];

                double a1 = y2 - y1, b1 = x1 - x2, c1 = a1 * x1 + b1 * y1;
                double a2 = y4 - y3, b2 = x3 - x4, c2 = a2 * x3 + b2 * y3;

                double det = a1 * b2 - a2 * b1;
                if (det != 0) {
                    double x = (b2 * c1 - b1 * c2) / det;
                    double y = (a1 * c2 - a2 * c1) / det;
                    intersections.emplace_back(static_cast<float>(x), static_cast<float>(y));
                    // cv2.circle(image_copy2, intersection, 5, (0, 255, 0), -1)
                }
            }
        }

        auto distance_between = [](const Point2f& p1, const Point2f& p2) {
            return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
        };

        const int cluster_threshold = 50;
        const int min_cluster_size = 80;

        if (!intersections.empty()) {
            vector<Point2f> clusters;
            unordered_set<int> used_points;

            for (size_t i = 0; i < intersections.size(); i++) {
                if (used_points.count(i)) continue;
                vector<Point2f> cluster = {intersections[i]};
                used_points.insert(i);

                for (size_t j = i + 1; j < intersections.size(); j++) {
                    if (used_points.count(j)) continue;
                    if (distance_between(intersections[i], intersections[j]) < cluster_threshold) {
                        cluster.emplace_back(intersections[j]);
                        used_points.insert(j);
                    }
                }

                if (cluster.size() > min_cluster_size) {
                    float avg_x = 0, avg_y = 0;
                    for (const auto& p : cluster) {
                        avg_x += p.x;
                        avg_y += p.y;
                    }
                    avg_x /= cluster.size();
                    avg_y /= cluster.size();
                    clusters.emplace_back(avg_x, avg_y);
                    // cv2.circle(image_copy2, point, 5, (0, 255, 255), -1)
                }
            }

            for (const auto& cluster : clusters) {
                circle(image_copy2, cluster, 5, Scalar(0, 255, 255), -1);
            }

            estimate_pose(clusters);
        }
    }

    // cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", image_copy2);
    return image_copy2;
}

float degrees(float radians) {
    return radians * 180 / CV_PI;
}

float radians(float degrees) {
    return degrees * CV_PI / 180;
}

int main() {
    std::cout << "ASAD's Code C++ VERSION\n";
    // Load camera calibration parameters
    try {
    FileStorage fs("calibration_params.yml", FileStorage::READ);
    if (!fs.isOpened())
{
    cout << "Could not open the calibration file: calibration_params.yml" << endl;
        return -1;
}
    fs["camMatrix"] >> camMatrix;
    fs["distCoef"] >> distCoef;
    fs.release();
    } catch (const cv::Exception& e) {
        cerr << "GOT ERROR LOADING CAMERA MATRIX" << endl;
        cerr << e.what() << endl;
        return 1;
    }

    // exit(0);

    std::cout << "Loaded Camera Matrix\n";

    // Calculate camera parameters
    FOCAL_LENGTH_PIXELS = camMatrix.at<double>(0, 0);
    IMAGE_WIDTH = camMatrix.at<double>(0, 2) * 2;
    HFOV = 2 * degrees(atan(IMAGE_WIDTH / (2 * FOCAL_LENGTH_PIXELS))); // 2 * atan((IMAGE_WIDTH / (2 * FOCAL_LENGTH_PIXELS))) * 180 / CV_PI;
    FOCAL_LENGTH_METERS = FOCAL_LENGTH_PIXELS / IMAGE_WIDTH;
    SENSOR_WIDTH = 2 * FOCAL_LENGTH_METERS * tan(radians(HFOV / 2));
    meters_to_pixels = FOCAL_LENGTH_PIXELS / SENSOR_WIDTH;
    pixels_to_meters = 1 / meters_to_pixels;

    string image_folder = "D:\Opencvmain\data 1";
    vector<string> image_paths;
    glob(image_folder + "*.png", image_paths, false);

    if (image_paths.empty()) {
        cout << "No images found in folder: " << image_folder << endl;
        return 1;
    }

    Mat first_image = imread(image_paths[0]);
    if (first_image.empty()) {
        cout << "Error reading images. Exiting..." << endl;
        return 1;
    }

    int height = first_image.rows, width = first_image.cols;

    namedWindow("Original and Filtered Footage", WINDOW_NORMAL);

    int frameCounter = 0;
    double frameTotalTime = 0;
    for (const auto& image_path : image_paths) {
        Mat original_image = imread(image_path);
        if (original_image.empty()) {
            cout << "Error reading image: " << image_path << endl;
            continue;
        }

        auto start = chrono::high_resolution_clock::now();
        Mat filtered_image = Filters(original_image);
        Mat combined_frame;
        hconcat(original_image, filtered_image, combined_frame);
        auto end = chrono::high_resolution_clock::now();
        frameTotalTime += chrono::duration_cast<chrono::nanoseconds>(end - start).count();

        imshow("Original and Filtered Footage", combined_frame);
        frameCounter++;

        if (waitKey(30) == 'q' || frameCounter >= 100) break;
    }

    cout << "Total Frames: " << frameCounter << " frames" << endl;
    cout << "Total Time: " << frameTotalTime / 1e9 << " seconds" << endl;
    cout << "Average Time: " << frameTotalTime / frameCounter / 1e9 << " seconds" << endl;
    cout << "FPS: " << frameCounter / (frameTotalTime / 1e9) << endl;

    return 0;
}
