import cv2
import numpy as np

gray_image = cv2.imread('./images/front_depth_1701437974_526695467.jpg', cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    print("Error: Could not open or find the image.")
else:
    # Access pixel values
    height, width = gray_image.shape
    print(f"Image dimensions: {width} x {height}")
    print(f"Pixel value at (0, 0): {gray_image[0, 0]}")

    # Image data is stored as a NumPy array
    print(type(gray_image))
    print(gray_image.shape)

    filtered_index = np.where(gray_image < 150)
    print(f"Min pixel value: {np.min(gray_image[filtered_index])}")
    print(f"Max pixel value: {np.max(gray_image[filtered_index])}")