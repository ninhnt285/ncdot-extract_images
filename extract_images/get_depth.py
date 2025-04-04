import rclpy
from rclpy.node import Node
import cv_bridge
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os

save_path = "/home/tnguy248/ros2_workspaces/ncdot/data/nov_17/17/test_depth"
Path(save_path).mkdir(parents=True, exist_ok=True)

class GetDepth(Node):
    def __init__(self):
        super().__init__('image_extractor')
        self.cv_bridge = cv_bridge.CvBridge()

        self.depth_sub = self.create_subscription(
            ImageMsg,
            '/zed_front/zed_node_0/depth/depth_registered',
            self.depth_callback,
            20
        )

        self.max_depth = []
        self.min_depth = []

    
    def depth_callback(self, msg: ImageMsg):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        timestamp = 1e-9 * nsec + sec
        timestamp_text = f'{sec:011}_{nsec:09}'

        try:
            # depth_data = self.cv_bridge.imgmsg_to_cv2(msg)
            depth_data = self.cv_bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error('Error converting ROS Image to OpenCV image: %s' % str(e))
            return
        
        # Rotate the image 180 degrees
        depth_data = cv2.rotate(depth_data, cv2.ROTATE_180)

        depth_data[np.isnan(depth_data)] = 10.0
        depth_data[np.isinf(depth_data)] = 10.0

        print(f"Saving depth_{timestamp_text}.")
        # image = Image.fromarray(depth_data)
        # image.save(os.path.join(save_path, f"depth_{timestamp_text}.tif"))

        depth_data = depth_data * 6500
        depth_data = depth_data.astype(np.uint16)
        cv2.imwrite(os.path.join(save_path, f"depth_{timestamp_text}.png"), depth_data)

        # self.min_depth.append(np.nanmin(depth_data))
        # self.max_depth.append(np.nanmax(depth_data))

        # print(np.nanmin(depth_data), ' - ', np.nanmax(depth_data), min(self.min_depth), max(self.max_depth))



def main(args=None):
    rclpy.init(args=args)

    depth_subscriber = GetDepth()

    rclpy.spin(depth_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depth_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()