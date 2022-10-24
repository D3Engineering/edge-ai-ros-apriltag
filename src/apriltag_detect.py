#!/usr/bin/env python3
import rospy
import sys
import cv2
import numpy as np
from d3_apriltag.msg import AprilTagDetection, AprilTagDetectionArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from apriltag import apriltag

class apriltag_detector:
	def __init__(self):
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/imx390/image_raw_rgb', Image, self.callback)
		self.tag_pub = rospy.Publisher('/apriltag_detections', AprilTagDetectionArray, queue_size=10)

	def callback(self, image):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
			detector = apriltag("tagStandard41h12")
			detections = detector.detect(cv_image)
			data = dict()
			logline = str(len(detections))+" AprilTags detected.\n"
			cv_result = cv_image
			for d in detections:
				corn = d['lb-rb-rt-lt']
				cv_result = cv2.polylines(cv_result, np.int32([np.array(corn)]), 1, (255,0,0), 2)
				cv_result = cv2.putText(cv_result, str(d['id']), (round(corn[3][0]), round(corn[3][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, 2)
				logline += "ID: "+str(d['id'])+", LB: "+str(np.int32(corn[0]))+", RB: "+str(np.int32(corn[1]))+", RT: "+str(np.int32(corn[2]))+", LT: "+str(np.int32(corn[3]))+"\n"
			rospy.loginfo(logline)
			global camera_info
			# TODO: Alex - this is where you have access to the 4 corners (among other AprilTag data) and the camera info
		except CvBridgeError as e:
			rospy.loginfo(e)

def main(args):
	rospy.init_node('apriltag_detector', anonymous=True)

	# Get camera name from parameter server
	global camera_name
	camera_name = rospy.get_param("~camera_name", "camera")
	camera_info_topic = "/{}/camera_info".format(camera_name)
	rospy.loginfo("Waiting on camera_info: %s" % camera_info_topic)

	# Wait until we have valid calibration data before starting
	global camera_info
	camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
	rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info.K))
	rospy.loginfo("Camera distortion coefficients: %s" % str(camera_info.D))
	april = apriltag_detector()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("Shutting Down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	print("Starting AprilTag Detector Node")
	main(sys.argv)
