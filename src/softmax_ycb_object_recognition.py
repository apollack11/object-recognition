import tensorflow as tf
import os
import cv2
import numpy as np

directory = "expo_black_dry_erase_marker/extracted_images"

files = []
for f in os.listdir(directory):
    if f.endswith('.jpg'):
        files.append(f)
    elif f.endswith('.png'):
        files.append(f)

width = 700
height = 700

num_images = len(files)

print "There are",num_images,"images in",directory

images = np.zeros((num_images, 784))
labels = np.zeros((num_images, 2))

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
    image = cv2.imread(directory + '/' + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.ravel()
    image = image.astype(float)
    image = 1 - image/255
    image = np.reshape(image, (1,784))
    # add image to images
    images[i] = image

    # import label
    digit = int(filename[0])
    image_label = np.array([0,0])
    image_label[digit] = 1
    image_label = np.reshape(image_label, (1,2))
    # # add image_label to labels
    labels[i] = image_label

### TENSORFLOW MODEL ###

# create a placeholder for input
# in this case we can input any number of MNIST images so None is used as the first element of the shape
x = tf.placeholder(tf.float32, [None, 784])

# create Vriables for the model parameters
W = tf.Variable(tf.zeros([784, 2]), 'W') # weights
b = tf.Variable(tf.zeros([2]), 'b') # biases

# implement the model
y = tf.matmul(x, W) + b

# to implement cross-entropy we need to add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 2])
# implement the cross-entropy function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# TensorFlow automatically does backpropagation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# operation to initilize Variables
init = tf.global_variables_initializer()

# now launch the model in a Session
sess = tf.Session()
sess.run(init)

# training (Using stochastic gradient descent)
for i in range(500):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs = images[i:i+1]
    batch_ys = labels[i:i+1]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# EVALUATING OUR MODEL
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# convert list of booleans to floating point numbers [1,0,1,1,etc.]
# take the mean of these predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print the accuracy of the model on our test data
print(sess.run(accuracy, feed_dict={x: images[500:609], y_: labels[500:609]}))

print(sess.run(y, feed_dict={x: images[500:609], y_: labels[500:609]}))

# save model
saver = tf.train.Saver()

save_path = saver.save(sess, "variables/softmax_model.ckpt")
print("Save to path: ", save_path)
