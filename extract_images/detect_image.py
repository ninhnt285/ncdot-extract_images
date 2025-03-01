import supervision as sv
import numpy as np


def detect_from_frame(model, frame, threshold=0.7):
    # -- get classes
    # dict maping class_id to class_name
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

    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]

    # convert to Detections
    detections = sv.Detections.from_ultralytics(results)

    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]

    # remove detections with confidence less than threshold
    detections = detections[detections.confidence >= threshold]
    # print (f"Confidence after trim: {detections.confidence}")

    # get confidence of remaining detections
    confidenceList = detections.confidence if len(detections) != 0 else []

    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]}"
        for class_id in detections.class_id
    ]

    # print ("Labels: ", labels)
    return detections, labels, confidenceList


def get_closest_distance_in_bounding_box(depth_frame, detections):
    # -- get the bounding box of the detection
    closest_distances = []
    if depth_frame is None:
        return closest_distances
    for detection in detections:
        x1, y1, x2, y2 = detection[0].astype(int)
        # get depth value of all pixels in the bounding box
        depth_values = depth_frame[y1:y2, x1:x2]
        # get the closest distance in the bounding box

        closest_distance = np.nanmin(depth_values)
        # if closest_distance > 10:
        #     closest_distance = np.inf

        closest_distances.append(closest_distance)

    return closest_distances
