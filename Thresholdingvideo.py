import cv2 as cv

def threshold_video(video_path, delay=30):
    # Open the video file
    cap = cv.VideoCapture(video_path)
    
    # Check if the video was loaded properly
    if not cap.isOpened():
        print("Error: Unable to open video. Please check the path.")
        return
    
    while True:
        # Read each frame
        ret, frame = cap.read()
        
        # Break the loop if no frames are left
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary_frame = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)
        cv.imshow("Thresholded Frame", binary_frame)
        
        # Exit the loop if the user presses the 'q' key
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

# Specify the path to the video file
video_path = r'D:\Opencv\applied.mp4'
threshold_video(video_path, delay=10)  # Adjust delay as needed
