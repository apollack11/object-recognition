#!/usr/bin/env python
from grip_node import GripperClient

import rospy
import image_geometry
from std_msgs.msg import Bool
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point
from baxter_core_msgs.msg import EndpointState, DigitalIOState
from object_recognition.msg import ObjectInfo
import planning_node as pnode
import baxter_interface
import tf

camera_state_info = None
camera_info = None

# THIS WORKS
def initCamera(data):
    global camera_info
    camera_info = data

# THIS WORKS
def getCameraState(data):
    global camera_state_info
    camera_state_info = data

def move_arm():
    print "Button was pressed."

    zinit = 0.00333971663214

    camera_x = camera_state_info.pose.position.x
    camera_y = camera_state_info.pose.position.y
    camera_z = camera_state_info.pose.position.z

    print "CAMERA X:",camera_x
    print "CAMERA Y:",camera_y
    print "CAMERA Z:",camera_z

    initial = [camera_x, camera_y, camera_z, 0.99, 0.01, 0.01, 0.01]

    pnode.initplannode(initial, "left")

    return

# THIS WORKS
def buttonPress(data):
    if data.state == 1:
        move_arm()

# def rotate_wrist(angle):
#     global object_angle
#
#     # Get desired joint values from parameter server
#     left_w0 = rospy.get_param('left_w0',default =0)
#     left_w1 = rospy.get_param('left_w1',default =0)
#     left_w2 = rospy.get_param('left_w2',default =0)
#     left_e0 = rospy.get_param('left_e0',default =0)
#     left_e1 = rospy.get_param('left_e1',default =0)
#     left_s0 = rospy.get_param('left_s0',default =0)
#     left_s1 = rospy.get_param('left_s1',default =0)
#
#     # Send the left arm to the desired position
#     rotated = {'left_w0': left_w0, 'left_w1': left_w1, 'left_w2': left_w2 + object_angle, 'left_e0': left_e0, 'left_e1': left_e1, 'left_s0': left_s0, 'left_s1': left_s1}
#     limb = baxter_interface.Limb('left')
#     limb.move_to_joint_positions(rotated)

# THIS WORKS
def arm_setup():
    # Get desired joint values from parameter server
    left_w0 = rospy.get_param('left_w0',default =0)
    left_w1 = rospy.get_param('left_w1',default =0)
    left_w2 = rospy.get_param('left_w2',default =0)
    left_e0 = rospy.get_param('left_e0',default =0)
    left_e1 = rospy.get_param('left_e1',default =0)
    left_s0 = rospy.get_param('left_s0',default =0)
    left_s1 = rospy.get_param('left_s1',default =0)

    # Send the left arm to the desired position
    home = {'left_w0': left_w0, 'left_w1': left_w1, 'left_w2': left_w2, 'left_e0': left_e0, 'left_e1': left_e1, 'left_s0': left_s0, 'left_s1': left_s1}
    limb = baxter_interface.Limb('left')
    limb.move_to_joint_positions(home)

if __name__ == '__main__':
    rospy.init_node('baxter_automated_image_capture', log_level=rospy.INFO)

    print "Moving arm to correct location"
    arm_setup()
    rospy.Subscriber("/cameras/left_hand_camera/camera_info", CameraInfo, initCamera)
    rospy.Subscriber("/robot/limb/left/endpoint_state", EndpointState, getCameraState)
    rate = rospy.Rate(50)
    while (camera_info is None) or (camera_state_info is None):
        rate.sleep()
    rospy.Subscriber("/robot/digital_io/left_button_ok/state", DigitalIOState, buttonPress)
    print "Ready to go!"
    rospy.spin()
