import os
import json
import sys
from collections import defaultdict

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
vehicle_cat = [2, 7, 3, 5]    # car, truck, motorcycle, or bus




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


vehicle_labels = ["car", "truck", "motorcycle", "bus"]

def analysis_data(dataset, prefix="front", working_dir=".", bag_index="00"):
    print(f"Processing {prefix} images")

    images = dataset[f"{prefix}_left"]
    processed_dir = f"{working_dir}/{bag_index}_processed/{prefix}"

    total_frame = len(images)
    vehicle_frame = 0
    person_frame = 0
    vehicle_frame_10 = 0
    person_frame_10 = 0
    distance_less_than_inf = defaultdict(lambda:[])


    #TODO: Loop over all exported json
    for timestamp in images:
        # check json file
        json_file = f"{processed_dir}/json/{prefix}_left_{timestamp}.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                has_vehicle = False
                has_vehicle_less_than_inf = False

                has_person = False
                has_person_less_than_inf = False


                # loop through data
                for dataRow in data:
                    label = dataRow['label']
                    distance = dataRow['closest_distance']
                    if distance != float('inf') and distance < 15 and distance > 0:
                        # Count the occurrences of distances
                        distance_less_than_inf[label].append(distance)
                        if label in vehicle_labels:
                            has_vehicle_less_than_inf = True
                            distance_less_than_inf["vehicle"].append(distance)

                        if label == "person":
                            has_person_less_than_inf = True

                    if label in vehicle_labels:
                        has_vehicle = True

                    if label == "person":
                        has_person = True

                
                vehicle_frame += (0 if not has_vehicle else 1)
                vehicle_frame_10 += (0 if not has_vehicle_less_than_inf else 1)
                
                person_frame += (0 if not has_person else 1)
                person_frame_10 += (0 if not has_person_less_than_inf else 1)

        else:
            # print(f"No matching .json file found for {jpg_file}")
            continue


    # # minimum distance of each label
    # for label, distances in distance_less_than_inf.items():
    #     if label not in ["vehicle", "person"]:
    #         continue

    #     min_distance = min(distances)
    #     print(f"Minimum distance for {label}: {min_distance:.02f} meters")
    # print ("-"*100)

    
    # # average distance of each label
    # for label, distances in distance_less_than_inf.items():
    #     if label not in ["vehicle", "person"]:
    #         continue

    #     avg_distance = sum(distances) / len(distances)
    #     print(f"Average distance for {label}: {avg_distance:02f} meters")
    # print ("-"*100)
    
    # Cate Distance
    for label in ["vehicle", "person"]:
        distances = distance_less_than_inf[label]
        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)
        print(f"Minimum distance for {label}: {min_distance:0.3f} meters")
        print(f"Average distance for {label}: {avg_distance:0.3f} meters")
        print ("-"*100)


    # percentage of frame
    print(f"Percentage of time at any distance for vehicle: {(vehicle_frame/total_frame*100):0.3f}%")
    print(f"Percentage of time distance is less than 10 meters for vehicle: {(vehicle_frame_10/total_frame*100):0.3f}%")

    print(f"Percentage of time at any distance for person: {(person_frame/total_frame*100):0.3f}%")
    print(f"Percentage of time distance is less than 10 meters for person: {(person_frame_10/total_frame*100):0.3f}%")
    print ("-"*100)

    # percentage of time distance is less than inf
    # for label, distances in distance_less_than_inf.items():
    #     less_than_inf = [d for d in distances if 0 < d < 15]
    #     percentage = len(less_than_inf) / total_frame * 100
    #     print(f"Percentage of time distance is less than 10 meters for {label}: {percentage:0.3f}%")
    # print ("-"*100)

    print("#"*100)



def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Usage: python readPathProcessJSON.py <directory>")
    #     sys.exit(1)
    # if sys.argv[1:]:
    #     directory = sys.argv[1]

    working_dir = "data/dec_18/09"
    bag_index = "01"

    if len(sys.argv) == 3:
        print("Usage: python readPathProcessJSON.py <directory> <bag_index>")
        working_dir = sys.argv[1]
        bag_index = sys.argv[2]
    

    dataset = load_json(f"{working_dir}/{bag_index}/timestamps.json")

    analysis_data(dataset, "front", working_dir, bag_index)
    analysis_data(dataset, "rear", working_dir, bag_index)
    # processDirectory(dataset, "front", working_path)
    # processDirectory(dataset, "rear", working_path)