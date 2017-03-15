import cv2
import numpy as np

# img = cv2.imread('messi5.jpg')
# res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# #OR
# height, width = img.shape[:2]
# res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

img = cv2.imread('nerf.jpg')
height, width = img.shape[:2]

aspect_ratio = float(width)/height

if width > height:
    if width > 640:
        width_new = 640
        height_new = int(width_new / aspect_ratio)
        res = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('nerf.jpg',res)

else:
    if height > 640:
        height_new = 640
        width_new = int(height_new * aspect_ratio)
        res = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('nerf.jpg',res)
