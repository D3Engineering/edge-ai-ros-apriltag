#!/usr/bin/env python3
import math
from json import JSONDecodeError
from typing import Union

import rospy
import sys
import cv2
import numpy as np
import tf
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from apriltag import apriltag
from scipy.spatial.transform import Rotation as R
import json
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class apriltag_odom:
    def __init__(self, num_frames: int, camera_info_topic_name: str, image_topic_name: str, camera_tf_name: str,
                 tag_tf_name: str):
        self.bridge = CvBridge()
        self.tfbr = tf.TransformBroadcaster()
        tfl = tf.TransformListener()
        self.num_frames = num_frames
        self.image_topic_name = image_topic_name
        self.camera_info_topic_name = camera_info_topic_name
        self.camera_tf_name = camera_tf_name
        self.tag_tf_name = tag_tf_name
        map_frame = "map"
        rospy.sleep(1)
        now = rospy.Time.now()
        tfl.waitForTransform(map_frame, self.tag_tf_name, now, rospy.Duration(4.0))
        self.tag_height = tfl.lookupTransform(map_frame, self.tag_tf_name, now)[0][2]
        # Wait until we have valid calibration data before starting
        rospy.loginfo("Waiting on camera_info: %s" % self.camera_info_topic_name)
        camera_info_msg = rospy.wait_for_message(self.camera_info_topic_name, CameraInfo)
        self.camera_info = np.array(camera_info_msg.K, dtype=np.float32).reshape((3, 3))
        #	camera_info = None
        rospy.loginfo("Camera intrinsic matrix: %s" % str(self.camera_info))

    def averageQuaternions(self, Q):
        # Number of quaternions to average
        M = Q.shape[0]
        A = np.zeros((4, 4))

        for i in range(0, M):
            q = Q[i, :]
            # multiply q with its transposed version q' and add A
            A = np.outer(q, q) + A

        # scale
        A = (1.0 / M) * A
        # compute eigenvalues and -vectors
        eigenValues, eigenVectors = np.linalg.eig(A)
        # Sort by largest eigenvalue
        eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
        # return the real part of the largest eigenvector (has only real part)
        return np.real(eigenVectors[:, 0].flatten())

    def get_instant_pose(self, image: Image) -> Union[Pose, None]:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
            detector = apriltag("tagStandard41h12")
            detections = detector.detect(cv_image)
            data = dict()
            logline = str(len(detections)) + " AprilTags detected.\n"
            for d in detections:
                corn = np.array(d['lb-rb-rt-lt'], dtype=np.float32)
                logline += "ID: " + str(d['id']) + ", LB: " + str(np.int32(corn[0])) + ", RB: " + str(
                    np.int32(corn[1])) + ", RT: " + str(np.int32(corn[2])) + ", LT: " + str(np.int32(corn[3])) + "\n"
                # Corner Refinement
                winsize = (5, 5)
                zerozone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
                corn_ref = corn
                # corn_ref = cv2.cornerSubPix(cv_image, corn, winsize, zerozone, criteria)
                logline += "[Refined] ID: " + str(d['id']) + ", LB: " + str(np.int32(corn_ref[0])) + ", RB: " + str(
                    np.int32(corn_ref[1])) + ", RT: " + str(np.int32(corn_ref[2])) + ", LT: " + str(
                    np.int32(corn_ref[3])) + "\n"

                tag_corners = np.array([corn_ref[0], corn_ref[1], corn_ref[3], corn_ref[2]], dtype=np.float32)
                tag_shape = (2, 2)
                tag_size = 0.3  # m
                # tag_size = 0.1042
                # Define target points. BL, BR, TL, TR
                objp = np.zeros((tag_shape[0] * tag_shape[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:tag_shape[0], 0:tag_shape[1]].T.reshape(-1, 2)
                objp = objp * tag_size

                # Find the rotation and translation vectors.
                iterations = 100  # Default 100
                reproj_error = 1.0  # Default 8.0
                confidence = 0.99  # Default 0.99
                pnp_flag = cv2.SOLVEPNP_ITERATIVE
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac( \
                    objp, tag_corners, self.camera_info, None, iterationsCount=iterations,
                    reprojectionError=reproj_error,
                    confidence=confidence, flags=pnp_flag)

                refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, sys.float_info.epsilon)
                rvecs, tvecs = cv2.solvePnPRefineLM(objp, tag_corners, self.camera_info, None, rvecs, tvecs,
                                                    refine_criteria)
                rospy.loginfo(tvecs)

                # Invert the Tag Pose to get the Robot's Pose
                rotmat = R.from_rotvec(rvecs.squeeze()).as_matrix()
                tfmat = np.hstack((rotmat, tvecs))
                tfmat = np.vstack((tfmat, [0, 0, 0, 1]))
                print("tfmat:")
                print(tfmat)
                res_tfmat = np.linalg.inv(tfmat)
                print("tfmat linalg:")
                print(res_tfmat)
                res_tvecs = res_tfmat[:3, 3]
                res_rotmat = res_tfmat[:3, :3]
                print("res_tvecs:")
                print(res_tvecs)
                print("res_rotmat:")
                print(res_rotmat)

                # Create a Robot Pose
                robot_pose_quat = R.from_matrix(res_rotmat).as_quat()
                robot_pose = Pose(Point(float(res_tvecs[0]), float(res_tvecs[1]), float(res_tvecs[2])),
                                  Quaternion(float(robot_pose_quat[0]), float(robot_pose_quat[1]),
                                             float(robot_pose_quat[2]), float(robot_pose_quat[3])))
                rospy.loginfo(logline)
                return robot_pose
            rospy.loginfo(logline)
            return None
        except CvBridgeError as e:
            rospy.loginfo(e)

    # Given a postiion & rotation (in quats) - returns a TransformStamped - ready for publishing
    def setup_static_transform(self, source, target, position, quaternion):
        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = source
        static_transformStamped.child_frame_id = target

        static_transformStamped.transform.translation.x = position[0]
        static_transformStamped.transform.translation.y = position[1]
        static_transformStamped.transform.translation.z = position[2]

        static_transformStamped.transform.rotation.x = quaternion[0]
        static_transformStamped.transform.rotation.y = quaternion[1]
        static_transformStamped.transform.rotation.z = quaternion[2]
        static_transformStamped.transform.rotation.w = quaternion[3]
        return static_transformStamped

    def get_pose(self) -> Pose:
        poses = []
        fails = 0
        while len(poses) < self.num_frames:
            print("Waiting on Frame " + str(len(poses) + 1) + "v" + str(fails) + "/" + str(self.num_frames))
            image = rospy.wait_for_message(self.image_topic_name, Image)
            result = self.get_instant_pose(image)
            if result is not None:
                poses.append(result)
                fails = 0
            else:
                fails += 1
        x_sum = 0
        y_sum = 0
        for pose in poses:
            x_sum += pose.position.x
            y_sum += pose.position.y
        x_avg = x_sum / self.num_frames
        y_avg = y_sum / self.num_frames
        poses_to_discard = []
        print("Average X: " + str(x_avg))
        print("Average Y: " + str(y_avg))
        for pose in poses:
            print("Actual X: " + str(pose.position.x))
            print("Actual Y: " + str(pose.position.y))
            print("X Diff: " + str(abs(abs(x_avg) - abs(pose.position.x))))
            print("Y Diff: " + str(abs(abs(y_avg) - abs(pose.position.y))))
            x_bad = abs(abs(x_avg) - abs(pose.position.x)) > 0.1
            y_bad = abs(abs(y_avg) - abs(pose.position.y)) > 0.1
            print("X Bad: " + str(x_bad))
            print("Y Bad: " + str(y_bad))
            if x_bad or y_bad:
                poses_to_discard.append(pose)
        for pose in poses_to_discard:
            poses.remove(pose)
        print("Removed " + str(len(poses_to_discard)) + " poses out of " + str(self.num_frames) + " poses")
        if len(poses) == 0:
            print("All poses were considered bad, trying pose estimation again...")
            return self.get_pose()
        x_sum, y_sum, z_sum = [0] * 3
        conversion_quaternions = []
        for pose in poses:
            x_sum += pose.position.x
            y_sum += pose.position.y
            z_sum += pose.position.z
            conversion_quaternions.append(
                (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z))
        plen = len(poses)
        avg_point = Point(x_sum / plen, y_sum / plen, self.tag_height)
        avg_quat_wxyz = self.averageQuaternions(np.array(conversion_quaternions, dtype=float))
        avg_quat_xyzw = Quaternion(avg_quat_wxyz[1], avg_quat_wxyz[2], avg_quat_wxyz[3], avg_quat_wxyz[0])
        (roll, pitch, yaw) = euler_from_quaternion([avg_quat_xyzw.x, avg_quat_xyzw.y, avg_quat_xyzw.z, avg_quat_xyzw.w])
        final_quat_arr = quaternion_from_euler(math.pi, 0, yaw)
        final_quat = Quaternion(final_quat_arr[0], final_quat_arr[1], final_quat_arr[2], final_quat_arr[3])
        print("X Final: " + str(avg_point.x))
        print("Y Final: " + str(avg_point.y))
        camera_pose = Pose(avg_point, final_quat)
        tag2camera_broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = self.setup_static_transform(self.tag_tf_name, self.camera_tf_name,
                                                              [avg_point.x, avg_point.y, avg_point.z],
                                                              final_quat_arr)
        tag2camera_broadcaster.sendTransform(static_transformStamped)
        # self.tfbr.sendTransform((avg_point.x, avg_point.y, avg_point.z),
        #                         (final_quat.x, final_quat.y, final_quat.z, final_quat.w),
        #                         rospy.Time.now(),
        #                         self.camera_tf_name,
        #                         self.tag_tf_name)
        return camera_pose


def tf_to_dict(tf):
    d = {
        "position": {
            "x": tf[0][0],
            "y": tf[0][1],
            "z": tf[0][2]
        },
        "orientation": {
            "x": tf[1][0],
            "y": tf[1][1],
            "z": tf[1][2],
            "w": tf[1][3]
        }
    }
    return d


def main(args):
    rospy.init_node('apriltag_detector', anonymous=True)
    camera_image_topic = "/imx390/image_raw_rgb"
    camera_info_topic = "/imx390/camera_info"
    camera_tf_name = "imx390_rear_optical"
    tag_tf_name = "apriltag21"

    running = True
    old_command = ""
    num_frames_str = ""
    base_path = "/opt/robotics_sdk/"
    tfl = tf.TransformListener()
    tfb = tf.TransformBroadcaster()
    print("AprilTag Pose Estimation")
    while not num_frames_str.isdigit():
        num_frames_str = input("Enter number of frames that should be taken to average per Pose Estimation: ")
    num_poses = int(num_frames_str)
    april = apriltag_odom(num_poses, camera_info_topic, camera_image_topic, camera_tf_name, tag_tf_name)
    save_file_name = ""
    while running and not rospy.is_shutdown():
        print("AprilTag Pose Estimation Commands: 'getpose', 'savepose', 'getposeloop', 'exit'")
        prompt_string = ""
        if old_command != "":
            prompt_string = "[" + old_command + "] > "
        else:
            prompt_string = "> "
        command = input(prompt_string)
        if command == "":
            if old_command != "":
                command = old_command
            else:
                continue
        old_command = command
        if command == "getpose":
            april.get_pose()
        elif command == "savepose":
            while save_file_name == "" or save_file_name == base_path:
                save_file_name = base_path + input("Enter name for Pose Save File: ")
            pose_name = ""
            while pose_name == "":
                pose_name = input("Enter name for Saved Pose: ")
            april.get_pose()
            rospy.sleep(2)
            lct = tfl.getLatestCommonTime("map", "base_link")
            robot_pose = tfl.lookupTransform("map", "base_link", lct)
            data = dict()
            try:
                new_file = open(save_file_name, "x+")
                print("File doesn't exist, creating...")
                new_file.write("{}")
                new_file.flush()
                new_file.close()
            except FileExistsError:
                print("File exists, loading...")
            with open(save_file_name, "r") as file:
                data = json.load(file)
                print("File loaded!")
            with open(save_file_name, "w") as file:
                data[pose_name] = tf_to_dict(robot_pose)
                print(data[pose_name])
                json.dump(data, file)
                print("File updated and saved! Path: " + save_file_name)
        elif command == "getposeloop":
            num_poses_str = ""
            while not num_poses_str.isdigit():
                num_poses_str = input("Enter number of iterations of Pose Estimation that should be looped through: ")
            num_poses = int(num_poses_str)
            captures = 0
            fails = 0
            while captures < num_poses:
                print("Waiting on Frame " + str(captures + 1) + "v" + str(fails) + "/" + str(num_poses))
                image = rospy.wait_for_message(camera_image_topic, Image)
                result = april.get_instant_pose(image)
                if result is not None:
                    captures += 1
                    fails = 0
                    tfb.sendTransform((result.position.x, result.position.y, result.position.z),
                                      (result.orientation.x, result.orientation.y, result.orientation.z,
                                       result.orientation.w),
                                      rospy.Time.now(),
                                      camera_tf_name,
                                      tag_tf_name)
                else:
                    fails += 1
        elif command == "exit":
            running = False
        else:
            print("Invalid Command")


if __name__ == '__main__':
    print("Starting AprilTag Detector Node")
    main(sys.argv)
