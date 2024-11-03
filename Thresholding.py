import cv2 as cv

def threshholding(image_path):
    image=cv.imread(image_path)
    if image is None:
        print("Error")
        return
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    cv.imshow("original image",image)
    cv.imshow("Threshold image",binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows

image_path = r'D:\Opencvmain\Auv\Opencv\enhanced_frame_004150.png'
threshholding(image_path)
