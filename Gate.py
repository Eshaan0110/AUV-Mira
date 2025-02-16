import cv2
import os

def images_to_video(images_folder, output_video_path, frame_rate=30):
    # Get all image file names in the folder
    images = [img for img in os.listdir(images_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    # Sort images to ensure they are in the correct order
    images.sort()
    
    # Check if there are any images in the folder
    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get frame dimensions
    first_image_path = os.path.join(images_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Loop through all images and write them to the video
    for image_name in images:
        image_path = os.path.join(images_folder, image_name)
        frame = cv2.imread(image_path)
        
        # Check if the frame was loaded properly
        if frame is None:
            print(f"Error: Unable to load image {image_name}. Skipping.")
            continue
        
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video_path}")

# Specify the folder containing images and the output video file path
images_folder = r'D:\Opencv\Auv\data2 applied'
output_video_path = r'D:\Opencv\Auv\data2 video'
images_to_video(images_folder, output_video_path)
