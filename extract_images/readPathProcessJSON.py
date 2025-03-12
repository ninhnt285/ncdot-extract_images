import os
import json
import sys
from collections import defaultdict



# Directory path
directory = 'dec18-2023_0825_frontleft'

def processDirectory(directory):
    jpg_files = []
    json_files = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            jpg_files.append(filename)
        elif filename.endswith('.json'):
            json_files.append(filename)

    # -- sort the jpg files
    jpg_files.sort()

    total_files = len(jpg_files)
    # instance_less_than_inf = defaultdict(lambda:0)
    distance_less_than_inf = defaultdict(lambda:[])
    # Loop through the .jpg files and check if there is a corresponding .json file
    for jpg_file in jpg_files:
        json_file = jpg_file[:-4] + '.json'  # Assuming the .json file has the same name as .jpg file
        if json_file in json_files:
            # print(f"Found matching .json file for {jpg_file}")
            json_file = os.path.join(directory, json_file)  # Append full path of directory to json_file
            # print (json_file)
            with open(json_file, 'r') as f:
                data = json.load(f)
                # print (data)
                # loop through data
                for dataRow in data:
                    # print (dataRow)
                    label = dataRow['label']
                    distance = dataRow['closest_distance']
                    if distance != float('inf') and distance != float('-inf'):
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
        less_than_inf = [d for d in distances if d < float('inf')]
        percentage = len(less_than_inf) / len(jpg_files) * 100
        print(f"Percentage of time distance is less than 10 meters for {label}: {percentage:0.3f}%")
    print ("-"*100)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python readPathProcessJSON.py <directory>")
        sys.exit(1)
    if sys.argv[1:]:
        directory = sys.argv[1]
    processDirectory(directory)