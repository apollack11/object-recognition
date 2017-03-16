import cv2
import numpy as np
import time

print cv2.__version__

# img = cv2.imread('nerf.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT()
# kp = sift.detect(gray,None)

# ##########
# ## based on train_classifier.py
# start = time.time()
#
# # Create feature extraction and keypoint detector objects
# feature_detector = cv2.FeatureDetector_create("SIFT")
# descriptor_extractor = cv2.DescriptorExtractor_create("SIFT")
#
# # List where all the descriptors are stored
# descriptor_list = []
# image = cv2.imread('soda_can_04.jpg')
# kp = feature_detector.detect(image)
# kp, descriptor = descriptor_extractor.compute(image, kp)
# descriptor_list.append(('soda_can_04.jpg', descriptor))
#
# image = cv2.drawKeypoints(image, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# print descriptor_list
#
# cv2.imwrite('sift_test.jpg', image)
#
# end = time.time()
#
# print ("Computation time: " + str(end - start))
# ## end
# ##########

image = cv2.imread('soda_can_test.jpg')
surf = cv2.SURF(400)
kp, des = surf.detectAndCompute(image, None)
print des.shape
print len(kp)
print surf.hessianThreshold
print surf.upright
print surf.descriptorSize()

image = cv2.drawKeypoints(image, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('surf_test2.jpg', image)
