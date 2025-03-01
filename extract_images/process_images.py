import os
import typing
import json
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import bisect
import datetime

from detect_image import detect_from_frame, get_closest_distance_in_bounding_box

model = YOLO("yolov9e.pt")

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
        distance_str = f'{distance:.2f}m' if distance != np.inf else '>10m'
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


def process_images(images, prefix="front", threshold=0.5):
    depth_images = images[prefix]

    count = 0
    for timestamp in images[f"{prefix}_left"][1:]:
        # Load image into cv2
        image_data = cv2.imread(
            f"{prefix}_left_{timestamp}.jpg", cv2.IMREAD_COLOR_RGB)

        # Load depth image
        try:
            idx = max(bisect.bisect_left(depth_images, timestamp) - 1, 0)
        except:
            idx = 0
        depth_data = cv2.imread(
            f"{prefix}_depth_{depth_images[idx]}.jpg", cv2.IMREAD_GRAYSCALE)
        depth_data = depth_data.astype(np.float32)
        depth_data = depth_data / 10.0

        # use yolov8 to detect cars
        detections, labels, confidences = detect_from_frame(
            model, image_data, threshold=0.5)

        if len(detections) > 0:
            closest_distances = get_closest_distance_in_bounding_box(
                depth_data, detections)
            # # -- convert unix_timestamp to datetime object in python
            # datetime_obj = datetime.datetime.fromtimestamp(unix_timestamp)
            # print("Received message at:", readable_time.strftime(
            #     "%Y-%m-%d%H:%M:%S:%f"), "at index: ", self.frameIndex)

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
                print(f"{l},{cd},{cv}", end=";")
            print()

            with open(f"./processed/{prefix}_left_{timestamp}.json", "w") as file_pointer:
                json.dump(json_detections, file_pointer)

            # -- save the image with distances
            save_image_with_distances(
                image_data,
                closest_distances,
                detections,
                labels,
                confidences,
                f"./processed/{prefix}_left_{timestamp}.jpg"
            )
        else:
            save_image_without_any_annotation(
                image_data, f"./processed/{prefix}_left_{timestamp}.jpg")

        count += 1
        if count == 1000:
            break


if __name__ == "__main__":
    images = load_json("timestamps.json")
    process_images(images, "front")
