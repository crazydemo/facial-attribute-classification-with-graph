#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:20:39 2018

@author: crazydemo
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import time
list_file  ="path/to/val.txt"
root = 'path/to/root'

count = 0
writer = tf.python_io.TFRecordWriter("vali.tfrecords")
with open(list_file, 'r') as f:
    for line in f:
        line = line.strip()
        field = line.split(' ')
        temp = field[1:41]
        label=[np.int(i) for i in temp]
        img = Image.open(root+field[0])
        if float(img.size[0])/float(img.size[1])>4 or float(img.size[1])/float(img.size[0])>4:
            continue
        img= img.resize((256,256))
        img_raw = img.tobytes()             
        example = tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                                                                       'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())
        count = count + 1
        if count%500 ==0:
            print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
    print "%d images are processed." %count
print 'Done!'  
writer.close()
