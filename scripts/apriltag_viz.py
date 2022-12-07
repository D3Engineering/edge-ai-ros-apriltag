#!/usr/bin/env python3
import rospy
import sys
import cv2
import numpy as np
from d3_apriltag.msg import AprilTagDetection, AprilTagDetectionArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped, Point, Quaternion
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

class apriltag_visualizer:
	def __init__(self, camera_info):
		self.camera_info = camera_info
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/imx390/image_raw_rgb', Image, self.image_callback)
		self.tag_sub = rospy.Subscriber('/apriltag_detections', AprilTagDetectionArray, self.apriltag_callback)
		self.image_pub = rospy.Publisher('/imx390/image_fused', Image, queue_size=10)
		self.current_image = None

	def image_callback(self, image):
		try:
			self.current_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
		except CvBridgeError as e:
			rospy.loginfo(e)

	def apriltag_callback(self, tags):
		this_image = self.current_image
		axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)
		axis = axis * 0.104 #0.104 = tag_size
		for tag in tags.detections:
			pose = tag.pose.pose.pose
			tvecs = np.array([pose.position.x, pose.position.y, pose.position.z])
			rvecs = np.array(R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_rotvec())
			this_image = project_and_draw_axis(this_image, axis, rvecs, tvecs, self.camera_info, None)
		try:
			image_message = self.bridge.cv2_to_imgmsg(this_image, encoding="bgr8")
			self.image_pub.publish(image_message)
		except CvBridgeError as e:
			print(e)

def main(args):
	rospy.init_node('apriltag_visualizer', anonymous=True)

	# Get camera name from parameter server
	camera_name = rospy.get_param("~camera_name", "camera")
	camera_info_topic = "/{}/camera_info".format(camera_name)
	rospy.loginfo("Waiting on camera_info: %s" % camera_info_topic)

	# Wait until we have valid calibration data before starting
	camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
	camera_info = np.array(camera_info.K, dtype=np.float32).reshape((3, 3))
	rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info))
	april = apriltag_visualizer(camera_info)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("Shutting Down")
	cv2.destroyAllWindows()
	
def draw_axis(img, corners, imgpts, linesize=2, colors=[(0,0,255), (0,255,0), (255,0,0)]):
    # If only one color is specified
    if np.shape(colors) != (3,3):
        colors = [colors] * 3

    corner = tuple(corners[0].ravel())
    # GBR colorspace
    # Use same color-axis convention as RVIZ, X-R, Y-G, Z-B
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), colors[0], linesize) # X - R
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), colors[1], linesize) # Y - G
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), colors[2], linesize) # Z - B
    return img

# Helper for drawing an axis with only a pose, projects the 2D origin point first
def project_and_draw_axis(img, axis, rvecs, tvecs, mtx, dist, linesize=2, colors=[(0,0,255), (0,255,0), (255,0,0)]):
    origin_pt, _ = cv2.projectPoints(np.float32([0,0,0]), rvecs, tvecs, mtx, dist)
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    return draw_axis(img, origin_pt, imgpts, linesize, colors)

if __name__ == '__main__':
	print("Starting AprilTag Visualizer Node")
	main(sys.argv)
