import tensorflow as tf
import numpy as np
import glob
from PIL import Image
#import PIL
#from PIL.Image import core as image
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
    img_resize(files)
   return y

def read_data(input_queue):
    label=input_queue[1]
    label=tf.reshape(label,[1,2])
    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_file,channels=3)
    #print(image.get_shape().as_list())
    image=tf.cast(image, tf.float32)
    return image, label

batch_size= tf.Variable(tf.constant(100))

testfiles =glob.glob('data/test/*.jpg')
test_labels=make_lab(testfiles)

testfiles = tf.convert_to_tensor(testfiles, tf.string)
test_labels = tf.convert_to_tensor(test_labels,tf.float32)

test_queue = tf.train.slice_input_producer([testfiles,test_labels])
testimg, labeltest = read_data(test_queue)
test_img, label_for_test =tf.train.batch ([testimg, labeltest],batch_size=batch_size, shapes=[[32,32,3], [1,2]])

test_img=tf.reshape(test_img,[batch_size,32,32,3])
label_for_test=tf.reshape(label_for_test,[batch_size,2])

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

x =  test_img
y_ =  label_for_test

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
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    
    tf.initialize_all_variables().run()
    saver.restore(sess, "lenet.ckpt")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    #for i in range(100):
    print(sess.run(y_conv))
    print(sess.run(accuracy))
    print(sess.run(b_conv1))
    
    coord.request_stop()
    coord.join(threads)
    sess.close()


