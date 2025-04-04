import cv2
import numpy as np
import os
import struct
from scipy import ndimage, misc
import matplotlib.pyplot

# def load_grayscale_image(image_name):
#     image = cv2.imread(image_name)
#     return image

# path = '/home/tnguy248/ros2_workspaces/ncdot/data/nov_17/17/00/front/depth'
path = "/home/tnguy248/ros2_workspaces/ncdot/data/nov_17/17/test_depth"
depth_min = []
depth_max = []

for filename in os.listdir(path):
    # print(f"Reading {filename}")
    # Read tif
    # gray_image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_ANYDEPTH)
    # Read png
    gray_image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_ANYDEPTH)
    gray_image = gray_image.astype(np.float32)
    gray_image = gray_image / 6500.0

    # print(gray_image.dtype)
    # gray_image = cv2.imread('./images/front_depth_1701437974_526695467.jpg', cv2.IMREAD_GRAYSCALE)
    # gray_image = gray_image.astype(np.float32)
    # gray_image = gray_image.view(dtype=float)

    if gray_image is None:
        print("Error: Could not open or find the image.")
    else:
        # Access pixel values
        # height, width = gray_image.shape
        # print(f"Image dimensions: {width} x {height}")
        # print(f"Pixel value at (0, 0): {gray_image[0, 0]}")

        # # Image data is stored as a NumPy array
        # print(type(gray_image))
        # print(gray_image.shape)

        filtered_index = np.where(gray_image < 10000)
        min_val = np.min(gray_image[filtered_index])
        max_val = np.max(gray_image[filtered_index])

        depth_min.append(min_val)
        depth_max.append(max_val)
        print(min_val, max_val, min(depth_min), max(depth_max))

        # print(f"Min pixel value: {np.min(gray_image[filtered_index])}")
        # print(f"Max pixel value: {np.max(gray_image[filtered_index])}")