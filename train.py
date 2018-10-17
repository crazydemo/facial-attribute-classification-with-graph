#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:35:19 2018

@author: crazydemo
"""

import tensorflow as tf 
import numpy as np
import os
from PIL import Image
from net import *
from get_data import *

BATCH_SIZE=128
VAL_BATCH_SIZE=128
num_of_attri = 40
total_step = 25600
pretrain = False
continue_train = False

last_model_path = 'path/to/your/restore/model'
train_file = 'path/to/your/train.tfrecords'
vali_file = 'path/to/your/vali.tfrecords'
tensorboard_path = 'path/logs/tensorboard'
model_path = 'path/logs/model/'
pretrain_path = 'path/to/your/pretrain/model'#Here we use the alexbn.npy model, please refer to readme for detail information.

'''The learning rate generator may be used, especially when training the lfwa dataset'''
def cyclical_learning_rate(global_step, step_size, max_bound, min_bound, decay, name=None):#cyclical learning rate generator
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    global_step = tf.cast(global_step, tf.float32)
    step_size = tf.convert_to_tensor(step_size, tf.float32)
    max_bound = tf.convert_to_tensor(max_bound, tf.float32)
    min_bound = tf.convert_to_tensor(min_bound, tf.float32)
    decay = tf.convert_to_tensor(decay, tf.float32)
    inverse = tf.floordiv(global_step, step_size)
    max_bound = tf.multiply(max_bound, tf.pow(decay, tf.floordiv(inverse, 2)))

    x = tf.mod(global_step, step_size)

    p = tf.cond(tf.mod(inverse, 2)<1, lambda: tf.divide((max_bound-min_bound), step_size), 
                              lambda: tf.divide((min_bound-max_bound), step_size))
    res = tf.multiply(x, p)
    return tf.cond(tf.mod(inverse, 2)<1, lambda: tf.add(res, min_bound, name), 
                              lambda: tf.add(res, max_bound, name))

def test_learning_rate(total_step, global_step, maximum_bound):#function for determining the hyparameters of cyclical learning rate 
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    global_step = tf.cast(global_step, tf.float32)
    max_bound = tf.convert_to_tensor(maximum_bound, tf.float32)
    k = tf.divide(max_bound, total_step)
    return tf.multiply(global_step, k)


'''basic definition and configuration'''
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

phase_train = tf.placeholder(tf.bool, name='phase_train')
x_image = tf.placeholder(tf.float32, [BATCH_SIZE, 227,227,3])
y = tf.placeholder(tf.int64, shape=[BATCH_SIZE, 40])
global_step=tf.Variable(0,trainable=False)
learning_rate = tf.train.polynomial_decay(0.001, global_step/3, 25600, 0)

lr_summary = tf.summary.scalar('learning_rate',learning_rate)

img,label = read_and_decode(train_file)
img_batch,label_batch = tf.train.shuffle_batch([img,label], batch_size=BATCH_SIZE, capacity=2000, min_after_dequeue=1000)
img_val,label_val = read_and_decode_test(vali_file)
img_batch_val,label_batch_val = tf.train.shuffle_batch([img_val,label_val], batch_size=VAL_BATCH_SIZE, capacity=2000, min_after_dequeue=1000)


'''net, loss, accuracy and summary'''
logits_cln, logits_fln, out1_summary = net(x_image, phase_train)

cross_entropy_cln = [0]*40
cross_entropy_flc = [0]*40                      
with tf.name_scope("cross_ent"):
    for i in range(num_of_attri):
        cross_entropy_cln[i] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:,i],logits=logits_cln[i]))
        cross_entropy_flc[i] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:,i],logits=logits_fln[i]))
cross_ent_cln_40 = tf.reduce_sum(cross_entropy_cln)
cross_ent_flc_40 = tf.reduce_sum(cross_entropy_flc)

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
reg_term_summary = tf.summary.scalar('reg_term', reg_term)
loss_cln = cross_ent_cln_40+reg_term
loss_fln = cross_ent_flc_40+reg_term
loss_cln_summary = tf.summary.scalar('loss_cln', loss_cln)
loss_fln_summary = tf.summary.scalar('loss_fln', loss_fln)

acc_cln = [0.0]*40
acc_fln = [0.0]*40
with tf.name_scope("accuracy"):
    for i in range(num_of_attri):
        temp_y_cln = tf.cast(tf.argmax(logits_cln[i],1),tf.int64)
        acc_cln[i] = tf.reduce_mean(tf.cast(tf.equal(temp_y_cln, y[:,i]), tf.float32))
        temp_y_fln = tf.cast(tf.argmax(logits_fln[i],1),tf.int64)
        acc_fln[i] = tf.reduce_mean(tf.cast(tf.equal(temp_y_fln, y[:,i]), tf.float32))
accuracy40_cln = tf.reduce_mean(acc_cln)
accuracy40_fln = tf.reduce_mean(acc_fln)
acc_cln_summary = tf.summary.scalar('acc_cln',accuracy40_cln)
acc_fln_summary = tf.summary.scalar('acc_fln',accuracy40_fln) 


merged_train_summary = tf.summary.merge([lr_summary, loss_cln_summary, loss_fln_summary, acc_cln_summary,acc_fln_summary, out1_summary])
merged_vali_summary = tf.summary.merge([loss_cln_summary, loss_fln_summary, acc_cln_summary, acc_fln_summary])

summary_writer_train = tf.summary.FileWriter(tensorboard_path+'/train', sess.graph)
summary_writer_test = tf.summary.FileWriter(tensorboard_path+'/test')

'''training method'''
var = tf.trainable_variables()
var_cln = []
var_pfln = []
var_backbone = []
for v in var:
    if "block" not in v.name and "cln" not in v.name:
        var_backbone.append(v)
    elif "block" in v.name and "cln" not in v.name:
        var_pfln.append(v)
    elif "cln" in v.name:
        var_cln.append(v)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_backbone = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_fln, var_list=var_backbone, global_step=global_step)
    train_cln = tf.train.MomentumOptimizer(learning_rate*10, 0.9).minimize(loss_tcln, var_list=var_cln, global_step=global_step)
    train_fln = tf.train.MomentumOptimizer(learning_rate*10, 0.9).minimize(loss_fln, var_list=var_fln, global_step=global_step)
train_op = tf.group([train_backbone, train_cln, train_fln])

sess.run(tf.global_variables_initializer())


'''main'''
if pretrain:
#############################alexbn##################################################
    print("pretrain...")
    weights_dict = np.load(pretrain_path, encoding='bytes').item()
    for op_name in weights_dict:
        if op_name not in ['fc6', 'fc7', 'fc8', 'fc6_bn', 'fc7_bn', 'fc8_bn']:
            with tf.variable_scope(op_name, reuse=True):
                data = weights_dict[op_name]
                if 'bn' in op_name:
                    var = tf.get_variable('beta', trainable=True)
                    sess.run(var.assign(data['mean']))
                    var = tf.get_variable('gamma', trainable=True)
                    sess.run(var.assign(data['variance']))
                else:
                    var = tf.get_variable('b', trainable=True)
                    sess.run(var.assign(data['biases']))
                    var = tf.get_variable('w', trainable=True)
                    sess.run(var.assign(data['weights']))
                print("restoring:"+op_name)
#############################alexbn##################################################
                
if continue_train:
    saver = tf.train.Saver(max_to_keep = None)
    saver.restore(sess, last_model_path)
    print("restoring...")

print("training...")
threads = tf.train.start_queue_runners(sess=sess)
for i in range(total_step):
    x_,y_= sess.run([img_batch,label_batch])
    op = [accuracy40_fln, accuracy40_cln, cross_ent_cln_40, cross_ent_fln_40, loss_cln, loss_fln, merged_train_summary, train_op, global_step]
    at_v, af_v, cet_v, cef_v, lt_v, lf_v, merged_train_summary_str, _, step=sess.run(op,feed_dict={x_image:x_, y: y_, phase_train:True})
    step /= 3
    print("step:{}, cet:{:.4f}, lt:{:.4f}, at:{:.4f}, cef:{:.4f}, lf:{:.4f}, af:{:.4f}\r".format(step, cet_v, lt_v, at_v, cef_v, lf_v, af_v))
    if step%25==0:
        summary_writer_train.add_summary(merged_train_summary_str,step)
    if step%100 == 0:
        x_,y_= sess.run([img_batch_val,label_batch_val])
        vali_op = [accuracy40_cln, accuracy40_fln, cross_ent_cln_40, cross_ent_fln_40, loss_cln, loss_fln, merged_tvali_summary]
        at_v, af_v, cet_v, cef_v, lt_v, lf_v, merged_vali_summary_str = sess.run(vali_op,feed_dict={x_image:x_, y: y_,phase_train:False})
        summary_writer_test.add_summary(merged_vali_summary_str,step)
        print("validating...")
        print("vali step:{}, cet:{:.4f}, lt:{:.4f}, at:{:.4f}, cef:{:.4f}, lf:{:.4f}, af:{:.4f}\r".format(step, cet_v, lt_v, at_v, cef_v, lf_v, af_v))
        print("training...")
    if step%100 == 0 and at_v>0.92:
        save_path = model_path + "model.ckpt"
        saver.save(sess,save_path,global_step=step)
    if step==total_step:
        save_path = model_path + "model.ckpt"
        saver.save(sess,save_path,global_step=step)
summary_writer_train.close()
summary_writer_test.close()
sess.close()
