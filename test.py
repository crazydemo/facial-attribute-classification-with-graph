#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:31:40 2018

@author: ivy
"""

import tensorflow as tf 
import numpy as np
from PIL import Image
from net import *
from get_data import *

TEST_BATCH_SIZE=1
num_of_attri = 40
total_num = 19962

test_file = "path/to/your/test.tfrecords"

phase_train = tf.placeholder(tf.bool, name='phase_train')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

x_image = tf.placeholder(tf.float32, [TEST_BATCH_SIZE, 227,227,3])                        
y = tf.placeholder(tf.int64, shape=[TEST_BATCH_SIZE, 40]) 

img_val,label_val= read_and_decode_test(test_file)
img_batch_val,label_batch_val = tf.train.batch([img_val,label_val], batch_size=VAL_BATCH_SIZE, capacity=2000)

logits, _, affinity_matrix= mynet(x_image, phase_train)
cross_entropy = [0]*40
with tf.name_scope("cross_ent"):
    for i in range(num_of_attri):
        cross_entropy[i] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:,i],logits=logits[i]))
cross_ent40 = tf.reduce_sum(cross_entropy)
acc = [0.0]*40
temp_y = [0]*40
with tf.name_scope("accuracy"):
    for i in range(num_of_attri):
        temp_y[i] = tf.cast(tf.argmax(logits[i],1),tf.int64)
        acc[i] = tf.reduce_mean(tf.cast(tf.equal(temp_y[i], y[:,i]), tf.float32))
accuracy40 = tf.reduce_mean(acc)

saver = tf.train.Saver(max_to_keep = None)

saver.restore(sess,'path/to/your/model')

mean_ce_ = 0.0
mean_acc_ = 0.0
acc_v = np.array([40])
acc_v_ = np.zeros([40])
visual_v = np.zeros([40, 40])

threads = tf.train.start_queue_runners(sess=sess)

for i in range(total_num):
    x_,y_= sess.run([img_batch_val,label_batch_val])
    vali_op = [cross_ent40, accuracy40, acc, logits, temp_y]
    vali_ce_v, vali_acc_v, acc_v, logits_v, y_predict_v, temp_y_v = sess.run(vali_op,feed_dict={x_image:x_, y:y_, phase_train:False})

    print("batch:{}, mean_ce:{:.4f}, mean_acc:{:.4f}".format(i, vali_ce_v, vali_acc_v))
    mean_ce_+=vali_ce_v
    mean_acc_+=vali_acc_v
    acc_v_ += acc_v
mean_ce_ /= total_num
mean_acc_ /= total_num
acc_v_ /= total_num
for i in range(40):
    print acc_v_[i]

print("mean_ce_:{:.4f}, mean_acc_:{:.4f}".format(mean_ce_, mean_acc_))

sess.close()