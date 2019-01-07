#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Defining the pad CNN

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email-me
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

def print_summary_model (tensor, first=False, last=False, verbose=True):
    if (verbose):
        if (first):
            print("\n------- CNN summary model -------\n")
        
        shape = tensor.shape.as_list()
        params = 1
        for v in shape:
            if (v is not None):
                params *= v

        print ("Layer: {}".format(tensor.name))
        print ("Shape: {} - Params: {}\n------------------".format(shape, params))
        
        if (last):
            print("\n------- End of CNN summary model -------\n")


def pad_2_net (dict_input, params):
    
    images = dict_input['images']
    n_labels = params['n_labels']
    verbose = params['verbose']


    if ('scalar_feat' in dict_input):
        scalar_feat = dict_input['scalar_feat']
    else:
        scalar_feat = None
    
    with tf.variable_scope('block_1'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=images, 
                filters=16, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, first=True, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)
    
    with tf.variable_scope('block_2'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=out, 
                filters=32, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)
        
    with tf.variable_scope('block_3'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=out, 
                filters=64, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)

    with tf.variable_scope('block_4'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=out, 
                filters=128, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)  

    with tf.variable_scope('block_5'):
        # Convolution and activation
        out = tf.layers.conv2d(
                inputs=out, 
                filters=128, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                activation=tf.nn.relu)
        
        print_summary_model (out, verbose=verbose)
        
        # pooling/subsampling
        out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2)
        print_summary_model (out, verbose=verbose)              
    
    
    with tf.variable_scope('block_fc1'):
        shape_out = out.get_shape().as_list()
        out = tf.reshape(out, [-1, shape_out[1] * shape_out[2] * shape_out[3]]) 
        
        # Feature aggregation
        if (scalar_feat is not None):
#            print (out.shape)
#            print (scalar_feat.shape)
            out = tf.concat([out, scalar_feat], 1)
#            print (out.shape)
        
        out = tf.layers.dense(inputs=out, units=128, activation=tf.nn.relu)
        print_summary_model (out, verbose=verbose)
        
    with tf.variable_scope('block_fc2'):
        logits = tf.layers.dense(inputs=out, units=n_labels, activation=tf.nn.relu)
        print_summary_model (logits, last=True, verbose=verbose)
        
    return logits