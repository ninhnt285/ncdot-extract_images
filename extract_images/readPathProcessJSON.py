import os
import json
import sys
from collections import defaultdict

def get_timestamps(timestamp: str) -> float:
    sec, nsec = timestamp.split("_")
    return float(f"{sec}.{nsec}")


def processDirectory(dataset, prefix="front", working_path="."):
    print(f"Processing {prefix} images")

    images = dataset[f"{prefix}_left"]
    images.sort()
    total_files = len(images)

    # instance_less_than_inf = defaultdict(lambda:0)
    distance_less_than_inf = defaultdict(lambda:[])

    last_time = get_timestamps(images[0])
    # Loop through the .jpg files and check if there is a corresponding .json file
    for timestamp in images[1:]:
        # image_file = f"{working_path}/{prefix}_processed/images/{prefix}_left_{timestamp}.jpg"
        json_file = f"{working_path}/{prefix}_processed/json/{prefix}_left_{timestamp}.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                # loop through data
                for dataRow in data:
                    label = dataRow['label']
                    distance = dataRow['closest_distance']
                    if distance != float('inf') and distance < 15:
                        # Count the occurrences of distances
                        distance_less_than_inf[label].append(distance)
        else:
            # print(f"No matching .json file found for {jpg_file}")
            continue

    # average distance of each label
    for label, distances in distance_less_than_inf.items():
        avg_distance = sum(distances) / len(distances)
        print(f"Average distance for {label}: {avg_distance:02f} meters")
    print ("-"*100)
    # minimum distance of each label
    for label, distances in distance_less_than_inf.items():
        min_distance = min(distances)
        print(f"Minimum distance for {label}: {min_distance:.02f} meters")
    print ("-"*100)
    # percentage of time distance is less than inf
    for label, distances in distance_less_than_inf.items():
        less_than_inf = [d for d in distances if d < 15]
        percentage = len(less_than_inf) / total_files * 100
        print(f"Percentage of time distance is less than 10 meters for {label}: {percentage:0.3f}%")
    print ("-"*100)

def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Usage: python readPathProcessJSON.py <directory>")
    #     sys.exit(1)
    # if sys.argv[1:]:
    #     directory = sys.argv[1]

    directory = "data/nov_17/17/"
    dataset = "00"

    working_path = directory + dataset

    if len(sys.argv) == 2:
        print("Usage: python readPathProcessJSON.py <directory>")
        directory = sys.argv[1]

    dataset = load_json(f"{working_path}/timestamps.json")

    processDirectory(dataset, "front", working_path)
    # processDirectory(dataset, "rear", working_path)