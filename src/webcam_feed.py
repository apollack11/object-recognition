#!/usr/bin/env python
import sys
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class webcam_image:
    def __init__(self):
        self.bridge = CvBridge()
        # webcam Subscriber
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
        # baxter camera Subscriber
        self.image_sub = rospy.Subscriber("/cameras/left_hand_camera/image",Image,self.callback)
        # Publisher
        # self.treasure_pub = rospy.Publisher("treasure_point",Point,queue_size=10)

    def callback(self,data):
        try:
            imgOriginal = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print("==[CAMERA MANAGER]==",e)

        blurred = cv2.GaussianBlur(imgOriginal,(11,11),0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # HSV values for red
        # lower = np.array([0,100,100])
        # upper = np.array([50,255,255])
        lower = np.array([0,0,255])
        upper = np.array([180,20,255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=7)
        mask = cv2.dilate(mask, None, iterations=7)
        output = cv2.bitwise_and(imgOriginal, imgOriginal, mask = mask)
        outputGrayscale = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        if major_ver == '3':
            contours = cv2.findContours(outputGrayscale,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
        elif major_ver == '2':
            contours = cv2.findContours(outputGrayscale,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

        print contours

        if len(contours) > 0:
            for c in contours:
                ((x,y),radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                contourCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 5:
                     cv2.circle(imgOriginal,(int(x),int(y)),int(radius),(0,255,0),2)
                     cv2.circle(imgOriginal,contourCenter,5,(0,0,255),-1)

        imgOriginal = cv2.flip(imgOriginal, 1)
        output = cv2.flip(output, 1)
        cv2.imshow("Image",imgOriginal)
        cv2.imshow("MyImage",output)
        cv2.waitKey(3)

def main(args):
    rospy.init_node('webcam_image', anonymous=True)
    ic = webcam_image()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    print "OpenCV Major Version:",major_ver
    main(sys.argv)
