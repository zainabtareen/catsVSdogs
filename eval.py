import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import glob


def make_lab(filenames):
   n=len(filenames)
   y = np.zeros((n,2), dtype = np.float32)
   for i, files in enumerate(filenames):      
   # If 'cat' string is in file name assign '1'
    if 'cat' in str(files):
     y[i,0] = 1
    else:
     y[i,1]=1
   return y

def read_data(input_queue):
    label=input_queue[1]
    label=tf.reshape(label,[1,2])
    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_file,channels=3)
    #print(image.get_shape().as_list())
    image.set_shape([50,50,3])
    image=tf.reshape(image,[1,7500])
    image=tf.cast(image, tf.float32)
    return image, label


testfiles =glob.glob('test/*.jpg')
test_labels=make_lab(testfiles)

testfiles = tf.convert_to_tensor(testfiles, tf.string)
test_labels = tf.convert_to_tensor(test_labels,tf.float32)

test_queue = tf.train.slice_input_producer([testfiles,test_labels])
testimg, labeltest = read_data(test_queue)
test_img, label_for_test =tf.train.batch ([testimg, labeltest],batch_size=100, shapes=[[1,7500], [1,2]])

test_img=tf.reshape(test_img,[100,7500])
label_for_test=tf.reshape(label_for_test,[100,2])

#softmax+

x2  =  test_img
y_2 =  label_for_test
W = tf.Variable(tf.zeros([7500, 2]),name="W")
b = tf.Variable(tf.zeros(shape=[2]),name="b")
y2 = tf.nn.softmax(tf.matmul(x2, W) + b)

saver = tf.train.Saver({'W':W,'b':b})


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver.restore(sess, "/tmp/model.ckpt")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_2,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #for i in range(100):
    print(sess.run(accuracy))
    #print(sess.run(y2))
    coord.request_stop()
    coord.join(threads)
    sess.close()


