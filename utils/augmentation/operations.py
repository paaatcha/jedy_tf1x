#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Author: André Pacheco
Email: pacheco.comp@gmail.com

This file contains some augmentation operations using tensorflow

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

'''
    Given an imagen this function returns the blur operation of this image
    Input:
        img: a TF image
    Output:
        blur_img: the image with blur operation
        
    Inspered by: https://github.com/chiralsoftware/tensorflow/blob/master/convolve-blur.py
'''
def blur_op (img):
    one_sixteenth = 1.0 / 16
    one_eighth = 1.0 / 8
    one_quarter = 1.0 / 4

    # We are taking a weighted average
    # of 3x3 pixels. Make sure the weights
    # add up to one
    filter_row_1 = [ 
      # in pixel (1,1)
      #   R             G              B
      [[ one_sixteenth, 0,             0],   # out channel R
       [ 0,             one_sixteenth, 0],   # out channel G
       [ 0,             0,             one_sixteenth ] ],  # out channel B
      
      # in pixel (2,1)
      #   R             G              B
      [ [ one_eighth,   0,             0],  # out channel R
       [ 0,             one_eighth,    0],   # out channel G
       [ 0,             0,             one_eighth ]],  # out channel B
      
      # in pixel (3,1)
      #   R             G              B
      [[ one_sixteenth, 0,             0],  # out channel R
       [ 0,             one_sixteenth, 0],  # out channel G
       [ 0,             0,             one_sixteenth ] ]  # out channel B
      ]
    
    filter_row_2 = [ 
      # in pixel (1,2)
      #   R             G              B
      [ [ one_eighth,   0,             0],  # out channel R
       [ 0,             one_eighth,    0],  # out channel G
       [ 0,             0,             one_eighth ] ],  # out channel B
      
      # in pixel (2,2)
      #   R             G              B
      [[ one_quarter,   0,             0],  # out channel R
       [ 0,             one_quarter,   0],   # out channel G
       [ 0,             0,             one_quarter ] ],  # out channel B
      
      # in pixel (3,2)
      #   R             G              B
      [ [ one_eighth,   0,             0],  # out channel R
       [ 0,             one_eighth,    0],  # out channel G
       [ 0,             0,             one_eighth ] ]  # out channel B
      ]
    
    filter_row_3 = [ 
      # in pixel (1,3)
      #   R             G              B
      [[ one_sixteenth, 0,             0],   # out channel R
       [ 0,             one_sixteenth, 0],   # out channel G
       [ 0,             0,             one_sixteenth ] ],  # out channel B
      
      # in pixel (2,3)
      #   R             G              B
      [ [ one_eighth,   0,             0],  # out channel R
       [ 0,             one_eighth,    0],  # out channel G
       [ 0,             0,             one_eighth ] ],  # out channel B
      
      # in pixel (3,3)
      #   R             G              B
      [[ one_sixteenth, 0,             0],  # out channel R
       [ 0,             one_sixteenth, 0],  # out channel G
       [ 0,             0,             one_sixteenth ]  # out channel B
       ]
      ]
    
    blur_filter = [filter_row_1, filter_row_2, filter_row_3] 
    
#    print (np.asarray(blur_filter))    
    
#    gaussian = [[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
#                    [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
#                    [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
#                    [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
#                    [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]
#        
#    zero = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
#    
#    blur_filter = tf.transpose(tf.constant([[gaussian,zero,zero],[zero,gaussian,zero],[zero,zero,gaussian]]))
    
#    blur_filter = tf.ones((3,3,3,3)) 
    
    
    batch = tf.stack([img])    
    blur_img = tf.nn.conv2d(batch, blur_filter, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True)    
    blur_img = tf.unstack(blur_img, num=1)[0]    
    return blur_img