import rclpy
from rclpy.node import Node
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
import numpy as np
from pathlib import Path

np.set_printoptions(threshold=np.inf)
np.core.arrayprint._line_width = 1000

"""
Topic: /zed_front/zed_node_0/left/image_rect_color | Type: sensor_msgs/msg/Image | Count: 9068 | Serialization Format: cdr
Topic: /zed_front/zed_node_0/right/image_rect_color | Type: sensor_msgs/msg/Image | Count: 9039 | Serialization Format: cdr
Topic: /zed_front/zed_node_0/depth/depth_registered | Type: sensor_msgs/msg/Image | Count: 9064 | Serialization Format: cdr
Topic: /zed_front/zed_node_0/imu/mag | Type: sensor_msgs/msg/MagneticField | Count: 29988 | Serialization Format: cdr
Topic: /zed_front/zed_node_0/imu/data | Type: sensor_msgs/msg/Imu | Count: 119884 | Serialization Format: cdr
Topic: /zed_front/robot_description | Type: std_msgs/msg/String | Count: 0 | Serialization Format: cdr

Topic: /zed_rear/zed_node_1/left/image_rect_color | Type: sensor_msgs/msg/Image | Count: 9064 | Serialization Format: cdr
Topic: /zed_rear/zed_node_1/right/image_rect_color | Type: sensor_msgs/msg/Image | Count: 9033 | Serialization Format: cdr
Topic: /zed_rear/zed_node_1/depth/depth_registered | Type: sensor_msgs/msg/Image | Count: 9063 | Serialization Format: cdr
Topic: /zed_rear/zed_node_1/imu/mag | Type: sensor_msgs/msg/MagneticField | Count: 30794 | Serialization Format: cdr
Topic: /zed_rear/zed_node_1/imu/data | Type: sensor_msgs/msg/Imu | Count: 119857 | Serialization Format: cdr
Topic: /zed_rear/robot_description | Type: std_msgs/msg/String | Count: 0 | Serialization Format: cdr


Topic: /tf_static | Type: tf2_msgs/msg/TFMessage | Count: 0 | Serialization Format: cdr
Topic: /tf | Type: tf2_msgs/msg/TFMessage | Count: 42273 | Serialization Format: cdr
Topic: /scan | Type: sensor_msgs/msg/LaserScan | Count: 6031 | Serialization Format: cdr
Topic: /velodyne_points | Type: sensor_msgs/msg/PointCloud2 | Count: 6031 | Serialization Format: cdr
Topic: /velodyne_packets | Type: velodyne_msgs/msg/VelodyneScan | Count: 6031 | Serialization Format: cdr


"""


class ImageExtractor(Node):
    def __init__(self):
        super().__init__('image_extractor')
        self.cv_bridge = cv_bridge.CvBridge()
        self.last_image_time = {}
        self.process_images = {}

        # Create directory to save images
        Path("./front/left").mkdir(parents=True, exist_ok=True)
        Path("./front/right").mkdir(parents=True, exist_ok=True)
        Path("./front/depth").mkdir(parents=True, exist_ok=True)

        Path("./rear/left").mkdir(parents=True, exist_ok=True)
        Path("./rear/right").mkdir(parents=True, exist_ok=True)
        Path("./rear/depth").mkdir(parents=True, exist_ok=True)


        depth_topics = {
            'front' : '/zed_front/zed_node_0/depth/depth_registered',
            'rear' : '/zed_rear/zed_node_1/depth/depth_registered'
        }
        for key in depth_topics:
            self.depth_sub = self.create_subscription(
                Image,
                depth_topics[key],
                lambda msg, key=key: self.depth_callback(msg, key),
                20
            )

        image_topics = {
            'front' : '/zed_front/zed_node_0/left/image_rect_color',
            'rear' : '/zed_rear/zed_node_1/left/image_rect_color',
        }
        for key in image_topics:
            self.image_sub = self.create_subscription(
                Image,
                image_topics[key],
                lambda msg, key=key: self.image_callback(msg, key),
                20
            )

    def depth_callback(self, msg: Image, camera: str = "front", threadhold=0.05):
        try:
            sec = msg.header.stamp.sec
            nsec = msg.header.stamp.nanosec
            timestamp = 1e-9 * nsec + sec
            timestamp_text = f'{sec:011}_{nsec:09}'

            if camera not in self.last_image_time.keys():
                self.last_image_time[camera] = timestamp
                self.process_images[camera] = []
            else:
                if timestamp - self.last_image_time[camera] < threadhold:
                    return
                self.last_image_time[camera] = timestamp
            self.process_images[camera].append(timestamp_text)

            depth_data = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Rotate the image 180 degrees
            depth_data = cv2.rotate(depth_data, cv2.ROTATE_180)
            # -- replace np.inf values with -1
            depth_data[np.isinf(depth_data)] = 20.0
            depth_data[np.isnan(depth_data)] = 20.0

            depth_data = depth_data * 10.0
            # filtered_index = np.where(depth_data < 200)
            # print(np.min(depth_data[filtered_index]), np.max(depth_data[filtered_index]))

            cv2.imwrite(f'{camera}/depth/{camera}_depth_{timestamp_text}.jpg', depth_data)

        except Exception as e:
            # self.get_logger().error('Error converting ROS Image to OpenCV image: %s' % str(e))
            return
        

    def image_callback(self, msg: Image, key: str = "front", side="left", threadhold=0.05):
        camera = f"{key}_{side}"

        try:
            sec = msg.header.stamp.sec
            nsec = msg.header.stamp.nanosec
            timestamp = 1e-9 * nsec + sec
            timestamp_text = f'{sec:011}_{nsec:09}'

            if camera not in self.last_image_time.keys():
                self.last_image_time[camera] = timestamp
                self.process_images[camera] = []
            else:
                if timestamp - self.last_image_time[camera] < threadhold:
                    return
                self.last_image_time[camera] = timestamp
            self.process_images[camera].append(timestamp_text)


            image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            temp_image = cv2.rotate(image, cv2.ROTATE_180)
            # -- get header data timestamp
            self.image_time = msg.header.stamp

            # -- remove the alpha channel
            if temp_image.shape[2] == 4:
                temp_image = temp_image[:, :, :3]

            cv2.imwrite(f'{key}/{side}/{camera}_{timestamp_text}.jpg', temp_image)

        except Exception as e:
            # self.get_logger().error('Error converting ROS Image to OpenCV image: %s' % str(e))
            return
        
def main(args=None):
    rclpy.init(args=args)
    extractor = ImageExtractor()
    print("Image extractor node started")

    try:
        while rclpy.ok():
            rclpy.spin(extractor)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
        pass

    print("Save timestamps to json file")
    import json
    with open('timestamps.json', 'w') as f:
        json.dump(extractor.process_images, f)

    extractor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()