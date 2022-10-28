#!/usr/bin/env python3
import rospy
import sys
import cv2
import numpy as np
import tf
from d3_apriltag.msg import AprilTagDetection, AprilTagDetectionArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped, Point, Quaternion
from nav_msgs.msg import Odometry
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from apriltag import apriltag
from scipy.spatial.transform import Rotation as R

class apriltag_detector:
	def __init__(self):
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/imx390/image_raw_rgb', Image, self.callback)
		self.tag_pub = rospy.Publisher('/apriltag_detections', AprilTagDetectionArray, queue_size=10)
		#self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=50)
		self.tfbr = tf.TransformBroadcaster()


	def callback(self, image):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
			detector = apriltag("tagStandard41h12")
			detections = detector.detect(cv_image)
			data = dict()
			logline = str(len(detections))+" AprilTags detected.\n"
			tag_detections = []
			for d in detections:
				corn = d['lb-rb-rt-lt']
				logline += "ID: "+str(d['id'])+", LB: "+str(np.int32(corn[0]))+", RB: "+str(np.int32(corn[1]))+", RT: "+str(np.int32(corn[2]))+", LT: "+str(np.int32(corn[3]))+"\n"
				global camera_info
				tag_corners = np.array([corn[0], corn[1], corn[3], corn[2]], dtype=np.float32)
				tag_shape = (2,2)
				tag_size = 0.3 # m

				# Define target points. BL, BR, TL, TR
				objp = np.zeros((tag_shape[0]*tag_shape[1],3), np.float32)
				objp[:,:2] = np.mgrid[0:tag_shape[0],0:tag_shape[1]].T.reshape(-1,2)
				objp = objp * tag_size

				# Find the rotation and translation vectors.
				iterations = 100  # Default 100
				reproj_error = 1.0 # Default 8.0
				confidence = 0.99  # Default 0.99
				pnp_flag = cv2.SOLVEPNP_ITERATIVE
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
				_, rvecs, tvecs, inliers = cv2.solvePnPRansac( \
				    objp, tag_corners, camera_info, None, iterationsCount=iterations, reprojectionError=reproj_error, confidence=confidence, flags=pnp_flag)

				refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, sys.float_info.epsilon)
				rvecs, tvecs = cv2.solvePnPRefineLM(objp, tag_corners, camera_info, None, rvecs, tvecs, refine_criteria)
				rospy.loginfo(tvecs)

				# Create a AprilTagDetection message
				tag_pose_quat = R.from_rotvec(rvecs.squeeze()).as_quat()
				tag_pose = Pose(Point(float(tvecs[0]), float(tvecs[1]), float(tvecs[2])), Quaternion(float(tag_pose_quat[0]), float(tag_pose_quat[1]), float(tag_pose_quat[2]), float(tag_pose_quat[3])))
				tag_pose_cov = PoseWithCovariance(tag_pose, [0]*36)
				tag_pose_header = std_msgs.msg.Header()
				tag_pose_header.stamp = rospy.Time.now()
				tag_pose_cov_stamp = PoseWithCovarianceStamped(tag_pose_header, tag_pose_cov)
				tag_detection = AprilTagDetection([d['id']], [tag_size], tag_pose_cov_stamp)
				tag_detections.append(tag_detection)

				# Invert the Tag Pose to get the Robot's Pose
				rotmat = R.from_rotvec(rvecs.squeeze()).as_matrix()
				tfmat = np.hstack((rotmat, tvecs))
				tfmat = np.vstack((tfmat, [0, 0, 0, 1]))
				print("tfmat:")
				print(tfmat)
				res_tfmat = np.linalg.inv(tfmat)
				print("tfmat linalg:")
				print(res_tfmat)
				res_tvecs = res_tfmat[:3,3]
				res_rotmat = res_tfmat[:3,:3]
				print("res_tvecs:")
				print(res_tvecs)
				print("res_rotmat:")
				print(res_rotmat)

				# Create a Robot Pose
				robot_pose_quat = R.from_matrix(res_rotmat).as_quat()
				#robot_pose = Pose(Point(float(res_tvecs[0]), float(res_tvecs[1]), float(res_tvecs[2])), Quaternion(float(robot_pose_quat[0]), float(robot_pose_quat[1]), float(robot_pose_quat[2]), float(robot_pose_quat[3])))

				# Create an Odometry message for the robot
				#odom = Odometry()
				#odom.header.stamp = rospy.Time.now()
				#odom.header.frame_id = "apriltag_"+str(d['id'])
				#odom.pose.pose = tag_pose
				#odom.pose.pose = robot_pose
				#self.odom_pub.publish(odom)

				# Broadcast Transform
				self.tfbr.sendTransform((res_tvecs[0], res_tvecs[1], res_tvecs[2]),
							(robot_pose_quat[0], robot_pose_quat[1], robot_pose_quat[2], robot_pose_quat[3]),
							rospy.Time.now(),
							"imx390_rear_optical",
							"apriltag21")
			rospy.loginfo(logline)
			tag_detections_header = std_msgs.msg.Header()
			tag_detections_header.stamp = rospy.Time.now()
			self.tag_pub.publish(AprilTagDetectionArray(tag_detections_header, tag_detections))

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
	camera_info = np.array(camera_info.K, dtype=np.float32).reshape((3, 3))
	rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info))
	april = apriltag_detector()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("Shutting Down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	print("Starting AprilTag Detector Node")
	main(sys.argv)
