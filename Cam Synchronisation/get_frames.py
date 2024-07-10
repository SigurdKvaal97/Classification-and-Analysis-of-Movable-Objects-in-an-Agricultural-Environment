import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def extract_images(bag_file, output_folder, image_topic):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file, "r")
    count = 0

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(f"{output_folder}/frame_{count:04d}.png", cv_img)
        count += 1

    bag.close()
    print(f"Extracted {count} frames to {output_folder}")

if __name__ == "__main__":
    bag_file = "/media/filip/T7 Shield/intelbags/M01_01.bag"  # replace with your .bag file
    output_folder = "intel_frames_M01_01"
    image_topic = "/device_0/sensor_1/Color_0/image/data"  # replace with your image topic

    extract_images(bag_file, output_folder, image_topic)
