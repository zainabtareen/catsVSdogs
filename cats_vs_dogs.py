import tensorflow as tf
import numpy as np
import os
from PIL import Image 
import glob

#                      ****** Preparing Data *******
STANDARD_SIZE = (50, 50)
def img_resize(filename, verbose=True):
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE,Image.ANTIALIAS)
    img.save(filename,quality=90)
    return

def img_to_matrix(filename, verbose=False):
        
    img = Image.open(filename)
    img = list(img.getdata())  #list makes it compatible for many operations
    img = map(list, img)  #converts each pixel into a list within the outer list
    img = np.array(img)   # each pixel's rgb value in a single row
    return img

def flatten_image(img):
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def data(filenames,data):
    n=len(filenames)
    print(n)
    y = np.zeros((n,2), dtype = np.float32)
    for i, files in enumerate(filenames):     
       if 'cat' in str(files):
          y[i,0] = 1
       else:
          y[i,1]=1
       img_resize(files)
       img = img_to_matrix(str(files))
       img = flatten_image(img)
       data.append(img)
    return data, y

filenames = glob.glob('*.jpg')
testfiles = glob.glob('test/*.jpg')
train_data = []
train_data,train_label=data(filenames,train_data)
train_data = np.array(train_data)
test_data =[]
test_data,test_label=data(testfiles,test_data)
test_data = np.array(test_data)

#*********************************************

#softmax

x = tf.placeholder(tf.float32, [None, 7500])
y_= tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.zeros([7500, 2]))
b = tf.Variable(tf.zeros(shape=[2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
l1= np.zeros([1,1])
l2= np.zeros([1,1])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y+ 1e-10), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(sess.run(cross_entropy,feed_dict={x: train_data, y_: train_label}))
for i in range(400):
 l1,l2=0,49
 for j in range(8):
   sess.run(train_step, feed_dict={x: train_data[l1:l2,:], y_: train_label [l1:l2,:]})
   l1=l1+50
   l2=l2+50
 print(sess.run(cross_entropy,feed_dict={x: train_data, y_: train_label}))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy,feed_dict={x:test_data, y_: test_label}))
print(sess.run(y,feed_dict={x:test_data}))

