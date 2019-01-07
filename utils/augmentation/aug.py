#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Author: André Pacheco
Email: pacheco.comp@gmail.com

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

from .operations import blur_op
import numpy as np
from numpy.random import rand, randint
from PIL import Image
from time import time

'''
    Given an image this function returns the augmentation according to a probability
    for this image
'''
def transform_img (img, flip_left_right=True, flip_up_down=True, 
                   crop=0.75, rot90=True, brightness=0.05, blur=True,
                   contrast=(0.7,0.9), hue=0.06, gamma=0.8,
                   saturation=(0.6,0.9), noise=(0.0,0.05), size=(256,256,3), seed_number=None):
    
    # Setting the seed if it exists
    np.random.seed(seed_number)
        
    if (flip_left_right):
        img = tf.image.random_flip_left_right(img, seed=seed_number)

    if (flip_up_down):
        img = tf.image.random_flip_up_down(img, seed=seed_number)
        
    if (brightness is not None):
        img = tf.image.random_brightness(img, brightness, seed=seed_number)
        
    if (contrast is not None):
        img = tf.image.random_contrast(img, contrast[0], contrast[1], seed=seed_number)
        
    if (saturation is not None):
        img = tf.image.random_saturation (img, saturation[0], saturation[1], seed=seed_number)
    
    if (hue is not None):
        img = tf.image.random_hue(img, hue, seed=seed_number)    

    if (gamma is not None and rand() > 0.5):
        img = tf.image.adjust_gamma(img, gamma)

    if (blur and rand() > 0.5):
        img = blur_op (img)
        
    if (crop is not None and rand() > 0.5):
        img = tf.image.central_crop(img, crop)
    
    if (rot90 and rand() > 0.5):
        img = tf.image.rot90(img, randint(1,3))
        
    if (noise is not None and rand() > 0.5):
        noise = tf.random_normal(shape=tf.shape(img), mean=noise[0], stddev=noise[1], dtype=tf.float32)
        img = tf.add(img, noise)
        
    if (size is not None):
        img = tf.image.resize_images(img, size[0:2])
        
    # Make sure the image is still in [0, 1]
    img = tf.clip_by_value(img, 0.0, 1.0)
        
    return img

'''
    This function just load an image or a list of images according to the path or
    paths set on entry
    
    Inputs:
        entry: a single or a list of paths of the images
        size: the size used to resize the image(s). If you'd like to use the original
        size, set it as None
        
    Output:
        A TF image format
'''
def load_img (entry, size=(256,256,3)):
    img = None
    img_list = None
        
    # Getting a group of images
    if (type(entry) == list):
        if (type(entry[0]) == str):
            img_list = list()
            for ent in entry:                
                img = tf.read_file(ent)
                img = tf.image.decode_jpeg(img, channels=size[2])
                img = tf.image.convert_image_dtype(img, tf.float32)
                if (size is not None):
                    img = tf.image.resize_images(img, size[0:2])
                img_list.append(img)
        else:
            img_list = entry
                
    # Getting a single image
    else:        
        if (type(entry) == str):            
            img = tf.read_file(entry)        
            img = tf.image.decode_jpeg(img, channels=size[2])
            img = tf.image.convert_image_dtype(img, tf.float32)
            if (size is not None):
                img = tf.image.resize_images(img, size[0:2])
                
        else:
            img = entry
        
    if (img_list is not None):        
        return img_list
    else:
        return img   

'''
    This function receives a list of paths and carry out the augmentation of these images. 
    It will save all augmentated images in the same folder as set in path
    
    Input:
        
    
'''
def save_batch_augmentation (paths, scalar_feat_ext=None, n_img_aug=None, flip_left_right=True, flip_up_down=True, 
                   crop=0.75, rot90=True, brightness=0.05, blur=True,
                   contrast=(0.7,0.9), hue=0.06, gamma=0.8,
                   saturation=(0.6,0.9), noise=(0.0,0.05), size=(256,256,3), seed_number=None, verbose=False):
    
    imgs = load_img(paths, size)
    n_imgs = len(imgs)
    
    if (n_img_aug is None):
        n_img_aug = n_imgs
    
    if (n_img_aug <= n_imgs):
        new_paths = paths[0:n_img_aug]
        new_imgs = imgs[0:n_img_aug]
    else:
        n_fac = n_img_aug // n_imgs
        n_rest = n_img_aug % n_imgs
        
        new_paths  = paths * n_fac
        new_imgs = imgs * n_fac
        if (n_rest > 0):
            new_paths = new_paths + paths[0:n_rest]
            new_imgs = new_imgs + imgs[0:n_rest]
    
    imgs = new_imgs
    paths = new_paths
    
    if (verbose):
        print ("Generating {} imagens augmented".format(len(imgs)))
    
    with tf.Session() as sess:
        for k in range(len(imgs)):
            imgs[k] = transform_img (imgs[k], flip_left_right, flip_up_down,
                crop, rot90, brightness, blur, contrast,
                hue, gamma, saturation, noise, size, seed_number)
            
            img = sess.run(imgs[k])
            
            base_name = paths[k].split('.')[0] + '_' + str(time()).replace('.','_')
            file_name = base_name + '.jpg'
            
            # Converting the image to 0-255 and convert to uint8
            img_rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
            im = Image.fromarray(img_rescaled)
            im.save(file_name)
            
            if (scalar_feat_ext is not None):
                path_feat = paths[k].split('.')[0] + '_feat.' + scalar_feat_ext
                new_path_feat = base_name + '_feat.' + scalar_feat_ext
                feat = np.loadtxt(path_feat)
                np.savetxt(new_path_feat, feat, fmt='%i', delimiter=',')
            
            if (verbose):
                print ("Image {} augmentaded and saved - {} of {}".format(file_name, k, len(imgs)))
            
        

#tf.reset_default_graph()
##
##path = ["/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited/TRAIN/carcinoma_baso_celular/700-2019-7138-0124_1527365585939.jpg", "/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited/TRAIN/carcinoma_baso_celular/700-3089-9949-3239_1533412062808.jpg", "/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited/TRAIN/melanoma/708-7041-2493-7893_1534513357093.jpg"]
#import sys
#sys.path.append('/home/labcin/CODIGOS/utils/')
#from utils_img import get_path_from_folders
#
#paths, _, _ = get_path_from_folders ('/home/labcin/AndrePacheco/Datasets/PAD/teste', shuf=False)
#
#print (paths)
#
#save_batch_augmentation (paths, n_img_aug=10, scalar_feat_ext="txt", verbose=True)

#shape = (256,256,3)
#
#img = tf.read_file(path)
#imgs = load_img(path)
#img = transform_img(imgs[2])

#print (img)

#img_decoded = tf.image.decode_jpeg(img, channels=shape[2])
#img = tf.image.convert_image_dtype(img_decoded, tf.float32)
#
#if (shape is not None):
#    img = tf.image.resize_images(img, shape[0:2])
#
#img = load_img(img)


#img_a = tf.image.flip_left_right(img)
#img_a = tf.image.flip_up_down(img)
# tf.image.random_flip_left_right(tf_img)
#img_a = tf.image.central_crop(img, 0.75)
#img_a = tf.image.rot90(img, 1)    
#img_a = tf.image.rot90(img, 2)
#img_a = tf.image.rot90(img, 3)
    
# igual flip
#img_a = tf.image.transpose_image (img)

#img_a = tf.image.adjust_brightness(img, -0.1)

#img_a = tf.image.random_brightness(img, 0.05)
    
#img_a = tf.image.adjust_contrast(img, 0.9)
#img_a = tf.image.random_contrast(img, 0.7, 0.9)
#img_a = tf.image.adjust_hue (img, -0.07)
#img_a = tf.image.random_hue(img, 0.06)

#img_a = tf.image.adjust_gamma(img, 0.8)
    
#img_a = tf.image.adjust_saturation(img, 0.7)
#img_a = tf.image.random_saturation (img, 0.6, 0.9)    
    
#img_a = tf.image.per_image_standardization(img)
    
# Adding Gaussian noise
#noise = tf.random_normal(shape=tf.shape(img), mean=0.0, stddev=0.05, dtype=tf.float32)
#img_a = tf.add(img, noise)


#img_a = blur_op(img)
#blur(img)

## Make sure the image is still in [0, 1]
#img_a = tf.clip_by_value(img_a, 0.0, 1.0)
#
#
#with tf.Session() as sess:
#    img_ = sess.run(img)
##    
##    img_a = sess.run(img_a)
##    
#    plt.imshow(img_)
##    
#    plt.show()
##    
##    print sess.run(noise)
#    
#    img_a = img_a.astype(np.uint8)
#    plt.imshow(img_a)
#    
#    plt.show()
#
#
#
#
