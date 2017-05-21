#!/usr/bin/env python
from grip_node import GripperClient

import rospy
import image_geometry
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point
from baxter_core_msgs.msg import EndpointState, DigitalIOState
from object_recognition.msg import ObjectInfo
import planning_node as pnode
import baxter_interface

camera_state_info = None
pixel_info = None
camera_info = None
button_pressed = False
global button_state
object_location = None

# def convertTo3D(pixel_info, camera_model, camera_x, camera_y, camera_z):
#
#     print "-----------------------------PIXEL--------------------------------"
#     print pixel_info.x
#     print pixel_info.y
#     val = camera_model.rectifyPoint([pixel_info.x, pixel_info.y])
#     print "-----------------------------VALUE--------------------------------"
#     print val
#     ray = camera_model.projectPixelTo3dRay(val)
#     # ray = camera_model.projectPixelTo3dRay(pixel_info)
#     print "-----------------------------RAY--------------------------------"
#     print ray
#     print "-----------------------------CAMERA--------------------------------"
#     print camera_x
#     print camera_y
#     print camera_z
#     xray_norm = ray[0]/ray[2] * (camera_z)*.9170
#     yray_norm = ray[1]/ray[2] * (camera_z)*1.536
#     x = (camera_x - yray_norm)
#     y = (camera_y - xray_norm)
#     print "-------------------------X Y -------------------------"
#     print (x,y)
#
#     return (x, y)

# THIS WORKS
def initCamera(data):
    global camera_info
    camera_info = data

# THIS WORKS
def getCameraState(data):
    global camera_state_info
    camera_state_info = data

# THIS WORKS
def getObjectLocation(data):
    global object_location
    # Wait until we have data about objects
    if data.names[0] != '':
        i = 0 # TODO: Replace this with "desired object"
        x = data.x[i]
        y = data.y[i]
        object_location = [x,y]
        # print "Object Location",object_location
        testnode()

def testnode():
    # print "----------------BUTTON VALUE----------"
    global button_state

    if button_state == 1:
        # button_pressed = False
        print "Button was pressed."
        # zsafe = -0.32 + 0.33 + 0.05
        # zdrop = -0.32 + 0.18 + 0.05
        # zpick = -0.32 + 0.13 + 0.05

        zsafe = -0.0370359868101
        zpick = -0.20

        camera_model = image_geometry.PinholeCameraModel()
        camera_model.fromCameraInfo(camera_info)
        # print "--------------CAMERA_MODEL--------------"
        # print camera_model
        gc = GripperClient()

        camera_x = camera_state_info.pose.position.x
        camera_y = camera_state_info.pose.position.y
        camera_z = camera_state_info.pose.position.z

        print camera_x
        print camera_y
        print camera_z

        zoffset = -0.28 # table height in baxter's frame (see setup_notes.txt)
        pixel_size = .0025 # camera calibration (meter/pixels)
        h = camera_z-zoffset # height from table to camera
        x0 = camera_x # x camera position
        y0 = camera_y # y camera position
        x_offset = 0 # offsets
        y_offset = -.02
        height = 400 # image frame dimensions
        width = 640
        cx = object_location[0]
        cy = object_location[1]
        print "Object Location (pixels):",(cx,cy)
        # Convert pixel coordinates to baxter coordinates
        xb = (cy - (height/2))*pixel_size*h + x0 + x_offset
        yb = (cx - (width/2))*pixel_size*h + y0  + y_offset

        print "Object Location (world):",(xb,yb)

        # des_pose = [xb, yb, -0.32+0.19+0.05, 0.99, 0.01, 0.01, 0.01]

        dsafe = [xb, yb, zsafe, 0.99, 0.01, 0.01, 0.01]
        dpick = [xb, yb, zpick, 0.99, 0.01, 0.01, 0.01]
        # ddropsafe = [0.75, -0.56, zsafe, 0.99, 0.01, 0.01, 0.01]
        # ddrop = [0.75, -0.56, zdrop, 0.99, 0.01, 0.01, 0.01]
        # leftsafe = [0.5, .5, zsafe, 0.99, 0.01, 0.01, 0.01]
        # rospy.sleep(2)

        print "Lets pick up the object"

        gc.command(position=100.0, effort=50.0)
        gc.wait()
        # rospy.sleep(2)

        pnode.initplannode(dsafe, "left")
        # rospy.sleep(0.1)

        # rospy.sleep(0.1)
        # pnode.initplannode(dpick, "left")
        # rospy.sleep(0.1)
        #
        # gc.command(position=40.0, effort=50.0)
        # gc.wait()
        #
        # pnode.initplannode(dsafe, "left")
        # rospy.sleep(0.1)
        # pnode.initplannode(dpick, "left")
        # rospy.sleep(0.1)
        #
        # print "Lets put the object back down"
        #
        # gc.command(position=100.0, effort=50.0)
        # gc.wait()
        #
        # arm_setup()

        return

# THIS WORKS
def buttonPress(data):
    global button_state
    button_state = data.state
    # if data.state == 1:
    #     button_pressed = True
    #     # testnode()
    #     rospy.sleep(1)

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
    print "Plan node is running"
    rospy.init_node('pickup_object', log_level=rospy.INFO)

    print "Moving arm to correct location"
    arm_setup()
    print "test1"
    rospy.Subscriber("/cameras/left_hand_camera/camera_info", CameraInfo, initCamera)
    print "test2"
    rospy.Subscriber("/robot/limb/left/endpoint_state", EndpointState, getCameraState)
    print "test3"
    rate = rospy.Rate(50)
    print "test4"
    while (camera_info is None) or (camera_state_info is None):
        rate.sleep()
    print "test5"
    rospy.Subscriber("/robot/digital_io/left_button_ok/state", DigitalIOState, buttonPress)
    rospy.Subscriber("/object_location", ObjectInfo, getObjectLocation)
    print "test6"
    rospy.spin()
