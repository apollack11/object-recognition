import tensorflow as tf
import os
import cv2

directory = "expo_black_dry_erase_marker"

files = []
for f in os.listdir(directory):
    if f.endswith('.jpg'):
        files.append(f)

width = 700
height = 700

for i,filename in enumerate(files):
    image_batch = filename[1]
    if image_batch == '1':
        x = 1925
        y = 1275
    if image_batch == '2':
        x = 1925
        y = 1000
    if image_batch == '3':
        x = 1925
        y = 1000
    if image_batch == '4':
        x = 1800
        y = 1100
    if image_batch == '5':
        x = 1800
        y = 1200
    if i % 100 == 0:
        print "Processed",i,"images"
    # import image and convert to numpy array of proper size
    fullImage = cv2.imread(directory + '/' + filename)
    extractedImage = fullImage[y:y+height, x:x+width]
    extractedImage = cv2.resize(extractedImage, (28, 28))
    newFile = 'extracted_images/0_%05d.jpg' %i
    cv2.imwrite(newFile,extractedImage)

# fullImage = cv2.imread("expo_black_dry_erase_marker/N5_294.jpg")
# extractedImage = fullImage[y:y+height, x:x+width]
# extractedImage = cv2.resize(extractedImage, (28, 28))
#
# cv2.imshow('extractedImage',extractedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
