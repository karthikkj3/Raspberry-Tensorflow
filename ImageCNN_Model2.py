# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:26:19 2018

@author: Dell
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_path = 'train.tfrecords'  # address to save the hdf5 file
def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 10, 10, 1], padding='SAME')

def max_pool_2x2(x):
	x = tf.nn.dropout(x, 0.85)
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

graph = tf.Graph()

with graph.as_default():
    
	# placeholders for input data batch_size x 750 x 750 x 3 and labels
	x = tf.placeholder(tf.float32, shape=[None, 750, 750, 3])
	y_ = tf.placeholder(tf.float32, shape=[None, 1])
    
	# defining decaying learning rate
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(1e-4, global_step=global_step, decay_steps=10000, decay_rate=0.97)
    
	# Conv Layer 1: with 16 filters of size 5 x 5 
	W_conv1 = weight_variable([25, 25, 3, 16])
	b_conv1 = bias_variable([16])

	x_image = tf.reshape(x, [-1, 750, 750, 3])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
	# Pool
	h_pool1 = max_pool_2x2(h_conv1)

	# Conv Layer 2:  with 32 filters of size 3 x 3 
	W_conv2 = weight_variable([19, 19, 16, 32])
	b_conv2 = bias_variable([32])
    
    
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
	# Pool
	h_pool2 = max_pool_2x2(h_conv2)
    
	# Conv Layer 3: with 64 filter of size 2 x 2
	W_conv3 = weight_variable([2, 2, 32, 64])
	b_conv3 = bias_variable([64])
    
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

	h_pool3 = max_pool_2x2(h_conv3)

	print("SHAPE", h_pool3.shape)
    
    
    
	W_fc1 = weight_variable([2*2*64, 1024])
	b_fc1 = bias_variable([1024])
    
	# flatening output of pool layer to feed in FC layer
	h_pool3_flat = tf.reshape(h_pool3, [-1, 2*2*64])
	print("SHAPE hpool3_flat", h_pool3_flat.shape)
	# FC layer
	h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

	# Dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 1])
	b_fc2 = bias_variable([1])

	# Output
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    #sess.run(tf.initialize_all_variables())
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    
    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [750, 750, 3])
    label = tf.reshape(label,[1])
    num_of_training_records = 0
    for record in  tf.python_io.tf_record_iterator(data_path):
       num_of_training_records +=1
    print(num_of_training_records)
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=4, capacity=30, num_threads=1, min_after_dequeue=10)
    
# Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    num_epoch = 2
    batch_size=4
    validation_error = []
    training_error = []
    for i in range(num_epoch):
        for j in range(0,num_of_training_records,batch_size):
        
            img, lbl = sess.run([images, labels])
            img = img.astype(np.float32)
            print(img)
            print(lbl)
            feed_dict = {x: img, y_: lbl, keep_prob: 0.5}
            if i%2 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:img, y_: lbl, keep_prob: 1.0})
                valid_accuracy = accuracy.eval(feed_dict={x:img, y_: lbl, keep_prob: 1.0})
                print("step: %d, training accuracy: %g, validation accuracy: %g" % 
					(i,train_accuracy, valid_accuracy ))
                training_error.append(1 - train_accuracy)
                validation_error.append(1 - valid_accuracy)
            train_step.run(feed_dict={x: img, y_: lbl, keep_prob: 0.5})
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()