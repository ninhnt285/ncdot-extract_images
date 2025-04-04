import os
import typing
import json
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import bisect
import datetime
import sys
from pathlib import Path
import time
import shutil

from detect_image import detect_from_frame, get_closest_distance_in_bounding_box

model = YOLO("yolov9c.pt")

CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
# 0: person
# 1: bicycle
# 2: car
# 3: motorcycle
# 5: bus
# 7: truck
# 9: traffic light
# 11: stop sign
# 36: skateboard

selected_classes = [0, 1, 2, 3, 5, 7, 9, 11, 36]


def convertForJSON(obj):
    # chat gpt suggested code
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    return obj


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def save_image_without_any_annotation(image, output_path):
    """ Save the image without any annotations of bounding boxes or distances
    """
    # -- save the image
    cv2.imwrite(output_path, image)


def save_image_with_distances(
    image,
    closest_distances,
    detections,
    labels,
    confidences,
    output_path
):
    # -- draw bounding boxes on the image
    idx = 0
    any_detection = True  # Make it false if only images with detections are to be saved
    for detection, distance, confidence in zip(detections, closest_distances, confidences):
        # if distance == np.inf:
        # idx += 1
        # continue
        any_detection = True
        x1, y1, x2, y2 = detection[0].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # -- put text and distance together
        distance_str = f'{distance:.2f}m' if distance < 10 else '>10m'
        confidenceStr = f'{confidence:.2f}'
        labelwConfidence = f"{labels[idx]}: {confidenceStr}"
        cv2.putText(image, labelwConfidence, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 1, cv2.LINE_4)
        cv2.putText(image, distance_str, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 1, cv2.LINE_4)
        idx += 1

    # -- save the image
    if any_detection:  # any detection within np.inf
        cv2.imwrite(output_path, image)


def process_images(images, prefix="front", working_dir=".", bag_index = "00", threshold=0.5):
    print(f"Processing {prefix} camera")
    start_time = time.time()

    original_path = f"{working_dir}/{bag_index}/{prefix}"
    processed_path = f"{working_dir}/{bag_index}_processed/{prefix}"

    # Create processed path
    Path(f"{processed_path}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{processed_path}/json").mkdir(parents=True, exist_ok=True)


    depth_images = images[prefix]
    depth_images.sort()

    count = 0
    for timestamp in images[f"{prefix}_left"]:
        # Load image into cv2
        image_data = cv2.imread(f"{original_path}/left/{prefix}_left_{timestamp}.jpg", cv2.IMREAD_COLOR_RGB)

        # Load depth image
        try:
            idx = max(bisect.bisect_left(depth_images, timestamp) - 1, 0)
        except:
            idx = 0
        # TODO: Need improvement in idx=0 case
        # print(f"{depth_images[max(idx-2, 0): min(idx+3, len(depth_images)-1)]} - {depth_images[idx]} - {timestamp}")
        depth_data = cv2.imread(
            f"{original_path}/depth/{prefix}_depth_{depth_images[idx]}.png", cv2.IMREAD_ANYDEPTH)
        depth_data = depth_data.astype(np.float32)
        depth_data = depth_data / 5000.0

        # use yolov to detect cars
        detections, labels, confidences = detect_from_frame(model, image_data, threshold)

        if len(detections) > 0:
            closest_distances = get_closest_distance_in_bounding_box(
                depth_data, detections)

            # -- format labels and closest distances
            json_detections = []
            for l, cd, cv in zip(labels, closest_distances, confidences):
                detection = {
                    "label": l,
                    "closest_distance": convertForJSON(cd),
                    "confidence": convertForJSON(cv),
                    "timestamp": convertForJSON(timestamp)
                }
                json_detections.append(detection)
                # -- print the detections and closest distances
                # print(f"{l},{cd},{cv}", end=";")
            # print()

            with open(f"{processed_path}/json/{prefix}_left_{timestamp}.json", "w") as file_pointer:
                json.dump(json_detections, file_pointer)

            # -- save the image with distances
            save_image_with_distances(
                image_data,
                closest_distances,
                detections,
                labels,
                confidences,
                f"{processed_path}/images/{prefix}_left_{timestamp}.jpg"
            )
        else:
            save_image_without_any_annotation(
                image_data, f"{processed_path}/images/{prefix}_left_{timestamp}.jpg")
        
        count += 1
        if count % 2000 == 0:
            print(f"Processed {count} images: {time.time() - start_time} seconds")
    print(f"Processed {count} images: {time.time() - start_time} seconds")


if __name__ == "__main__":
    working_dir = "data/nov_20/13"
    bag_index = "03"

    if len(sys.argv) == 3:
        print("Usage: python process_images.py <directory> <bag_index>")
        working_dir = sys.argv[1]
        bag_index = sys.argv[2]

    dataset = load_json(f"{working_dir}/{bag_index}/timestamps.json")

    process_images(dataset, "front", working_dir, bag_index=bag_index)
    process_images(dataset, "rear", working_dir, bag_index=bag_index)

    # Copy the timestamps.json file to the processed directory
    shutil.copyfile(
        f"{working_dir}/{bag_index}/timestamps.json",
        f"{working_dir}/{bag_index}_processed/timestamps.json"
    )
    
