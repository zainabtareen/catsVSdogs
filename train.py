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

#epoch     = tf.Variable(100)

batch_size= tf.Variable(tf.constant(400))

filenames = glob.glob('*.jpg')
labels    =make_lab(filenames)

filenames = tf.convert_to_tensor(filenames, tf.string)
labels = tf.convert_to_tensor_or_indexed_slices(labels,tf.float32)

train_queue =tf.train.slice_input_producer([filenames,labels], shuffle=True)
train_img, label = read_data(train_queue)
train_batch, label_batch =tf.train.batch ([train_img,    label],batch_size=batch_size, shapes=[[1,7500], [1,2]])

train_batch=tf.reshape(train_batch,[batch_size,7500])
label_batch=tf.reshape(label_batch,[batch_size,2])

#softmax+
x   =  train_batch
y_  =  label_batch
W = tf.Variable(tf.zeros([7500, 2]),name="W")
b = tf.Variable(tf.zeros(shape=[2]),name="b")
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver({'W':W,'b':b})

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y+ 1e-10), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(cross_entropy)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run(cross_entropy))
    for i in range(500):
     sess.run(train_step)
     print(sess.run(cross_entropy))
    
    save_path = saver.save(sess, "/tmp/model.ckpt")
    coord.request_stop()
    coord.join(threads)
    sess.close()


