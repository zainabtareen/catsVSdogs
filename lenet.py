import tensorflow as tf
#from tensorflow.python.framework import ops
#from tensorflow.python.framework import dtypes
import numpy as np
from PIL import Image
#import Image
import glob
import math
STANDARD_SIZE = (32, 32)
def img_resize(filename, verbose=True):
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE,Image.ANTIALIAS)
    img.save(filename,quality=90)
    return
def make_lab(filenames):
   n=len(filenames)
   y = np.zeros((n,2), dtype = np.float32)
   for i, files in enumerate(filenames):      
   # If 'cat' string is in file name assign '1'
    if 'cat' in str(files):
     y[i,0] = 1
    else:
     y[i,1]=1
    #img_resize(files)
   return y

def read_data(input_queue):
    label=input_queue[1]
    label=tf.reshape(label,[1,2])
    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_file,channels=3)
    #img=tf.image.resize_images(image,32,32)
    #img=tf.reshape(img,[1,3072])
    img=tf.cast(image, tf.float32)
    return img, label

#epoch     = tf.Variable(100)

batch_size= tf.Variable(tf.constant(400))

filenames = glob.glob('data/*.jpg')
labels    =make_lab(filenames)

filenames = tf.convert_to_tensor(filenames, tf.string)
labels = tf.convert_to_tensor_or_indexed_slices(labels,tf.float32)

train_queue =tf.train.slice_input_producer([filenames,labels], shuffle=True)
train_img, label = read_data(train_queue)
train_batch, label_batch =tf.train.batch ([train_img,    label],batch_size=batch_size, shapes=[[32,32,3], [1,2]])
#, shapes=[[1,3072], [1,2]]
train_batch=tf.reshape(train_batch,[batch_size,32,32,3])
label_batch=tf.reshape(label_batch,[batch_size,2])

#softmax+
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

x   =  train_batch
y_  =  label_batch

W_conv1 = weight_variable([5, 5, 3, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([5 * 5 * 16, 120])
b_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([120, 84])
b_fc2 = bias_variable([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

W_out = weight_variable([84, 2])
b_out= bias_variable([2])
y_out = tf.matmul(h_fc2,W_out) + b_out
y_conv=tf.nn.softmax(y_out)

saver = tf.train.Saver()
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv+1e-10), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run(cross_entropy))
    for i in range(20):
     sess.run(train_step)
     print(sess.run(cross_entropy))
    save_path = saver.save(sess, "lenet.ckpt")
    coord.request_stop()
    coord.join(threads)
    sess.close()
