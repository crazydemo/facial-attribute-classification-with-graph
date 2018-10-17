#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:30:09 2018

@author: crazydemo
"""

import tensorflow as tf 
from PIL import Image

image_height=227
image_width=227

def read_and_decode(filename):   
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([40], tf.int64), 
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label

def read_and_decode_test(filename):   
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([40], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label

def read_and_decode_test_ordinal(filename):   
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([40], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string), 
                                                                    'name': tf.FixedLenFeature([], tf.int64)})
    name = tf.cast(features['name'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label,name