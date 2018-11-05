#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:21:08 2018

@author: crazydemo
"""

import tensorflow as tf
import tensorflow.python.layers.layers

REGULARIZATION_RATE = 0.001
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

def max_pool(x, kheight, kwidth, stridex, stridey, padding):
  return tf.nn.max_pool(x, ksize=[1, kheight, kwidth, 1],strides=[1, stridex, stridey, 1], padding=padding)

def bn(x, phase_train, name, activation=None):
    if activation=="relu":
        activation = tf.nn.relu
    return tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True, is_training=phase_train,scope=name)

def conv(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding = "SAME"):
    channel = int(x.get_shape()[-1])
    shape = [kHeight, kWidth, channel, featureNum]
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=shape, initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        b = tf.get_variable('b', shape = [featureNum], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d(x, w, strides=[1, strideX, strideY, 1], padding=padding)+b
    return out

def positionFeature(x):
    pool = tf.reduce_mean(x, 3)
    out = tf.expand_dims(pool, 3)
    return out

def sigmLayer(x):
    return tf.nn.sigmoid(x)

def gapLayer(x, kHeight, kWidth, padding = "VALID"):
    return tf.nn.avg_pool(x, ksize = [1, kHeight, kWidth, 1], strides = [1, 1, 1, 1], padding = padding)

def fc(x, outD, name):
    inD = int(x.get_shape()[-1])
    layer_flat = tf.reshape(x, [-1, inD])
    shape = [inD, outD]
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=shape, initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        b = tf.get_variable('b', shape = [outD], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(layer_flat, w) + b
    return out

def fc_conv(x, outD, name):
    h = int(x.get_shape()[1])
    w = int(x.get_shape()[2])
    c = int(x.get_shape()[3])
    inD = h*w*c
    layer_flat = tf.reshape(x, [-1, inD])
    shape = [inD, outD]
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=shape, initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        b = tf.get_variable('b', shape = [outD], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(layer_flat, w) + b
    return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convolution"""
    channel = int(x.get_shape()[-1])
    initializer = tf.contrib.layers.xavier_initializer()
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum], initializer=initializer)
        b = tf.get_variable("b", shape = [featureNum], initializer=initializer)

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
      
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return out


def net(x, train_phase):
    y_predict_cln = [0]*40
    y_predict_fln = [0]*40
    
    data_bn = bn(x, train_phase, "data_bn")
#    print('bn0:{}'.format(bn0.get_shape()))
    conv1 = conv(data_bn, 11, 11, 4, 4, 96, "conv1", "VALID")
    conv1_bn = bn(conv1, train_phase, "conv1_bn", "relu")
    pool1 = max_pool(conv1_bn, 3, 3, 2, 2, "VALID")
#    print('pool1:{}'.format(pool1.get_shape()))
    conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", "SAME", 2)
    conv2_bn = bn(conv2, train_phase, "conv2_bn", "relu")
    pool2 = max_pool(conv2_bn, 3, 3, 2, 2, "VALID")
#    print('pool2:{}'.format(pool2.get_shape()))
    conv3 = conv(pool2, 3, 3, 1, 1, 384, "conv3")
    conv3_bn = bn(conv3, train_phase, "conv3_bn", "relu")
#    print('bn3:{}'.format(bn3.get_shape()))
    conv4 = convLayer(conv3_bn, 3, 3, 1, 1, 384, "conv4", "SAME", 2)
    conv4_bn = bn(conv4, train_phase, "conv4_bn", "relu")
#    print('bn4:{}'.format(bn4.get_shape()))
    conv5 = convLayer(conv4_bn, 3, 3, 1, 1, 256, "conv5", "SAME", 2)
    conv5_bn = bn(conv5, train_phase, "conv5_bn", "relu")
    pool5 = max_pool(conv5_bn, 3, 3, 2, 2, "VALID")
#    print('pool5:{}'.format(pool5.get_shape()))
    
    with tf.variable_scope("block0") as scope:
            bconv1 = conv(pool5, 1, 1, 1, 1, 256, "bconv1")
            bbn1 = bn(bconv1, train_phase, "bbn1", "relu")
#            print('bbn1:{}'.format(bbn1.get_shape()))
            pf = positionFeature(bbn1)
#            print('pf:{}'.format(pf.get_shape()))
            bconv2 = conv(pf, 1, 1, 1, 1, 16, "bconv2")
            bbn2 = bn(bconv2, train_phase, "bbn2", "relu")
#            print('bbn2:{}'.format(bbn2.get_shape()))
            bconv3 = conv(bbn2, 1, 1, 1, 1, 1, "bconv3", "VALID")
            sigm3 = sigmLayer(bconv3)
#            print('sigm3:{}'.format(sigm3.get_shape()))
            ele_mul = tf.multiply(bbn1, sigm3)
######################      40 tasks     ########################################################      
            gap = gapLayer(ele_mul, 6, 6)
            fc_out = fc(gap, 2, scope.name)
            y_predict_fln[0] = fc_out
######################      40 tasks     ########################################################

            psetemp = conv(ele_mul, 1, 1, 1, 1, 10, "cln_conv", "SAME")
            pseout = psetemp
#            print('pseout:{}'.format(pseout.get_shape()))
            b, h, w, c = psetemp.get_shape().as_list()
            affinity_matrix = tf.reshape(psetemp, [b, h*w*c, 1])
#            print('affinity_matrix:{}'.format(affinity_matrix.get_shape()))

    for i in range(1,40):
        with tf.variable_scope("block"+str(i)) as scope:
            bconv1 = conv(pool5, 1, 1, 1, 1, 256, "bconv1")
            bbn1 = bn(bconv1, train_phase, "bbn1", "relu")
            
            pf = positionFeature(bbn1)
#            print('pf:{}'.format(pf.get_shape()))
            bconv2 = conv(pf, 1, 1, 1, 1, 16, "bconv2")
            bbn2 = bn(bconv2, train_phase, "bbn2", "relu")
#            print('bbn2:{}'.format(bbn2.get_shape()))
            bconv3 = conv(bbn2, 1, 1, 1, 1, 1, "bconv3", "VALID")
            sigm3 = sigmLayer(bconv3)
#            print('sigm3:{}'.format(sigm3.get_shape()))
            ele_mul = tf.multiply(bbn1, sigm3)
######################      40 tasks     ########################################################      
            gap = gapLayer(ele_mul, 6, 6)
            fc_out = fc(gap, 2, scope.name)
            y_predict_fln[i] = fc_out
######################      40 tasks     ########################################################
            psetemp = conv(ele_mul, 1, 1, 1, 1, 10, "cln_conv", "SAME")
            pseout = tf.concat([pseout, psetemp], 3)
             b, h, w, c = psetemp.get_shape().as_list()
            affinity_matrix = tf.concat([affinity_matrix, tf.reshape(psetemp, [b, h*w*c, 1])], 2)
            
    affinity_matrix_ = tf.matmul(tf.transpose(affinity_matrix, [0, 2, 1]), affinity_matrix)
    merge_mat = tf.expand_dims(affinity_matrix_, 3)
    with tf.variable_scope("affinity_mat") as scope:
        tf.summary.image('merge_mat', merge_mat)
#    print('affinity_matrix_:{}'.format(affinity_matrix_.get_shape()))
    affinity_weight = [0]*40
    affinity_weight = tf.split(affinity_matrix_,40, 1)
#    print('affinity_weight[0]:{}'.format(affinity_weight[0].get_shape()))
    pseout_ = tf.reshape(pseout, [b, h*w*c, 40])
#    print('pseout_:{}'.format(pseout_.get_shape()))
    pseout_ = tf.transpose(pseout_, [0,2,1])
    for i in range(40):
            with tf.variable_scope('cln_fc'+str(i)) as scope:
                aw = tf.nn.softmax(affinity_weight[i])
                tf.summary.histogram("aw", aw)
                weighted_pse = tf.matmul(aw,pseout_)
                fc_out = fc(weighted_pse, 2, scope.name)
                y_predict_cln[i] = fc_out
    merged_summary =  tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'affinity_mat')])
    return y_predict_cln, y_predict_fln, merged_summary
            
        
