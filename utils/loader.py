# -*- coding: utf-8 -*-
'''
This file has some auxiliary functions to load and handle images

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find some bug, please email-me
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from random import shuffle, seed
import glob
import tensorflow as tf
import numpy as np
import os
from .augmentation.aug import transform_img



'''
    This function binarizes a vector (one hot enconding)
    For example:
        Input: v = [1,2,3]
        Output: v = [1,0,0;
                     0,1,0;
                     0,0,1]
    
        Input:
            ind: a array 1 x n
            N: the number of indices. If None, the code get is from the shape
        Output:
            The one hot enconding array n x N    
'''
def one_hot_encoding(ind, N=None):
    ind = np.asarray(ind)
    if ind is None:
        return None
    
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)

'''     
    This function returns a 2 list: a list of folders' name in a root folder and a list of
    all images' path in all folders
    For example: if we have the following tree:
        IMGs
            - A
                img1.png
                img2.png
            - B
                img3.ong
                img4.png
    The root folder is IMGs, its children will be A, B. paths will return a list composed by 
    IMGs/{children}/img{number}.png and fold_names = ['A', 'B']
    
    Input:
        path: root folder path
        img_ext: the image extension
        shuf: if you'd like to shuffle the list of paths set it as True
        scalar_feat_ext: the extension of a file of scalar feature
    Output:
        paths: a list of images' paths in all folders
        fold_names: a list of name of all folders' in the root folder    
'''
def get_path_from_folders (path, scalar_feat_ext=None, img_ext='jpg', shuf=True):
    paths = list()      
    fold_names = [nf for nf in os.listdir(path) if os.path.isdir(os.path.join(path, nf))]  
    scalar_feat = list()
            
    if (len(fold_names) == 0):
        folders = glob.glob(path)
        fold_names = [path.split('/')[-1]]
    else:
        folders = glob.glob(path + '/*')
    
    for fold in folders:                   
        paths += (glob.glob(fold+'/*.'+img_ext))            
    
    if (shuf):        
        shuffle(paths)    
        
    if (scalar_feat_ext is not None):
        for p in paths:
            scalar_feat.append( np.loadtxt(p.split('.')[0]+'_feat.'+scalar_feat_ext, dtype=np.float32) )
            #scalar_feat_paths.append(p.split('.')[0]+'_feat.'+scalar_feat)
    
    return paths, np.asarray(scalar_feat), fold_names 

'''    
    This gets a list of images' path and get all labels from the inner folder each image is inside.
    For example: aaa/bbb/ccc/img.png, the label will be ccc. Each path will have its own label

    Input:
        path: root folder path
        n_samples: number of samples that you wanna load from the path list
        img_ext: the image extension
        shuf: if you'd like to shuffle the list of paths set it as True
        one_hot: if you'd like the one hot encoding set it as True
        scalar_feat_ext: the extension of a file of scalar feature
    Output:
        paths: a list of images' paths in all folders
        labels: a list of the labels related with the paths
        dict_labels: a python dictionary relating path and label

'''
def get_path_and_labels_from_folders (path, scalar_feat_ext=None, n_samples=None, img_ext='jpg', shuf=False, one_hot=True):
    labels = list()
    
    # Getting all paths
    paths, scalar_feat, folds = get_path_from_folders (path, scalar_feat_ext, img_ext, shuf)    
    dict_labels = dict()
    
    
    value = 0
    for f in folds:
        if (f not in dict_labels):
            dict_labels[f] = value
            value += 1
    
    if (n_samples is not None):
        paths = paths[0:n_samples]
    
    for p in paths:
        lab = p.split('/')[-2]
        labels.append(dict_labels[lab])
        
    if (one_hot):                
        labels = one_hot_encoding(labels)
    else:
        labels = np.asarray(labels)
        
    return paths, labels, scalar_feat, dict_labels  


'''
    It gets a python dictionary and returns the list of train, val and test sets
    
    Input:
        data_dict: a dictionary cotaining the image path as key and the label and scalar feature as values
        sets_perc: the set percentage for [train, val, test]. It must sum up 1.0
        shuf: wether shuffle the dataset
        seed_number: the seed to shuffle
        one_hot: whether using one hot encoding
        
    Output:
        img: the tensor with the loaded image
        label: the image label
'''
def get_path_and_labels_from_dict (data_dict, sets_perc=None, shuf=True, seed_number=None, one_hot=True):
       
    img_path_list = data_dict.items()
    dict_labels = dict()
    
    # Generating the labels to get the labels in numbers
    value = 0
    for lab in [item[1][1] for item in img_path_list]:
        if (lab not in dict_labels):
            dict_labels[lab] = value
            value += 1
    
    print (dict_labels)
    
    # Getting the sets partition (train, val and test)
    if (sets_perc is None):
        val_perc = 0.1
        test_perc = 0.1
        train_perc = 0.8
    else:
        s = sum(sets_perc)
        if (abs(1.0-s) >= 0.01):
            print ("The sets percentage must sum up 1. The sum was {}".format(s))
            raise ValueError
        else:
            train_perc = sets_perc[0]
            val_perc = sets_perc[1]
            test_perc = sets_perc[2]
    
            
    if (shuf):
        # This is used to keep the same partitions for each train, val and test sets
        if (seed_number is not None):
            seed(seed_number)
        shuffle(img_path_list)    
        
    n_samples = len(img_path_list)    
    
    if (val_perc == 0.0): # In this case, there's no val set
        n_train = int(round(n_samples * train_perc))
        n_test = n_samples - n_train
        n_val = 0
        
        train_img_path_list = img_path_list[0:n_train]
        test_img_path_list = img_path_list[n_train:n_samples]
        val_img_path_list = None        
        
    else:
        n_train = int(round(n_samples * train_perc))
        n_test = int(round(n_samples * test_perc))
        n_val = n_samples - n_train - n_test
        
        train_img_path_list = img_path_list[0:n_train]
        test_img_path_list = img_path_list[n_train:n_train+n_test]
        val_img_path_list = img_path_list[n_train+n_test:n_samples]
        
    
    
    print ("\n# of train samples: {}".format(n_train))
    print ("# of val samples: {}".format(n_val))
    print ("# of test samples: {}\n".format(n_test))
    
    if (one_hot):
        train_list = [[item[0] for item in train_img_path_list], [item[1][0] for item in train_img_path_list], one_hot_encoding([dict_labels[item[1][1]] for item in train_img_path_list])]       
        if (val_img_path_list is not None):
            val_list = [[item[0] for item in val_img_path_list], [item[1][0] for item in val_img_path_list], one_hot_encoding([dict_labels[item[1][1]] for item in val_img_path_list])]       
            
        test_list = [[item[0] for item in test_img_path_list], [item[1][0] for item in test_img_path_list], one_hot_encoding([dict_labels[item[1][1]] for item in test_img_path_list])]       
    else:
        train_list = [[item[0] for item in train_img_path_list], [item[1][0] for item in train_img_path_list], [dict_labels[item[1][1]] for item in train_img_path_list]]       
        if (val_img_path_list is not None):
            val_list = [[item[0] for item in val_img_path_list], [item[1][0] for item in val_img_path_list], [dict_labels[item[1][1]] for item in val_img_path_list]]       
            
        test_list = [[item[0] for item in test_img_path_list], [item[1][0] for item in test_img_path_list], [dict_labels[item[1][1]] for item in test_img_path_list]]       
    
    return train_list, val_list, test_list

'''
    It gets an image path and returns a tensor with the image loaded and its related label
    
    Input:
        path: the image path
        label: the image label
        size: a tupla with the a new width and height. If you don't wanns chenge the image size
              set it as None
        channels: the image's depth  
        scalar_feat: if you're also loading scalar features with the images, you
        should use this parameter
        
    Output:
        img: the tensor with the loaded image
        label: the image label
'''
def load_img_as_tensor (path, label, size=(128,128), channels=3, scalar_feat=None, root_folder=None):
    img = tf.read_file(path)

    if (root_folder is None):
        # Don't use tf.image.decode_image, or the output shape will be undefined
        img_decoded = tf.image.decode_jpeg(img, channels=channels)
    else:
        img_decoded = tf.image.decode_jpeg(root_folder + '/' + img, channels=channels)
        
    # This will convert to float values in [0, 1]
    img = tf.image.convert_image_dtype(img_decoded, tf.float32)

    # Image resizing. This is very important to get the tensor shape in the model
    if (size is not None):        
        img = tf.image.resize_images(img, size)

    if (scalar_feat is not None):
        return img, scalar_feat, label
    else:
        return img, label
    

'''
    It gets an tensor with an image loaded and runs some augmentation operations
    
    Input:
        image: the tensor with the loaded image
        label: the image label
        scalar_feat: if you're also loading scalar features with the images, you
        should use this parameter
        params: a dictionary representing the augmentation params. If None, it set it as defaults.
            flip_left_right = True 
            flip_up_down = True
            crop = 0.75
            rot90 = True
            brightness = 0.05
            blur = True
            contrast = (0.7,0.9)
            hue = 0.06
            gamma = 0.8
            saturation = (0.6,0.9)
            noise = (0.0,0.05)
            size = (256,256,3)
            seed_number = None
'''
def get_aug_tf(image, label, scalar_feat=None, params=None):
    
    if (params is None):
        flip_left_right = True 
        flip_up_down = True
        crop = 0.75
        rot90 = True
        brightness = 0.05
        blur = True
        contrast = (0.7,0.9)
        hue = 0.06
        gamma = 0.8
        saturation = (0.6,0.9)
        noise = (0.0,0.05)
        size = (128,128,3)
        seed_number = None
    else:
        if ('flip_left_right' in params.keys()):
            flip_left_right = params['flip_left_right']
        else:
            flip_left_right = False
        
        if ('flip_up_down' in params.keys()):
            flip_up_down = params['flip_up_down']
        else:
            flip_up_down = False
        
        if ('crop' in params.keys()):        
            crop = params['crop']
        else:
            crop = None
            
        if ('rot90' in params.keys()):
            rot90 = params['rot90']
        else:
            rot90 = False
            
        if ('brightness' in params.keys()):
            brightness = params['brightness']
        else:
            brightness = None
            
        if ('blur' in params.keys()):
            blur = params['blur']
        else:
            blur = False
            
        if ('contrast' in params.keys()):
            contrast = params['contrast']
        else:
            contrast = None
            
        if ('hue' in params.keys()):
            hue = params['hue']
        else:
            hue = None
            
        if ('gamma' in params.keys()):            
            gamma = params['gamma'] 
        else:
            gamma = None
            
        if ('saturation' in params.keys()):
            saturation = params['saturation']
        else:
            saturation = None
            
        if ('noise' in params.keys()):            
            noise = params['noise']
        else:
            noise = None
            
        if ('size' in params.keys()):            
            size = params['size']
        else:
            size = None
        
        if ('seed_number' in params.keys()):
            seed_number = params['seed_number']
        else:
            seed_number = None

    image = transform_img (image, flip_left_right, flip_up_down, crop,
                           rot90, brightness, blur, contrast, hue, gamma,
                           saturation, noise, size, seed_number)
    
    if (scalar_feat is not None):
        return image, scalar_feat, label
    else:
        return image, label

'''
    It gets as parameter a list of paths and labels and returns the dataset according to tf.data.Dataset.
    
    Input:
        paths: the list of imagens path
        labels: the labels for each image in the path's list
        is_train: set True if this dataset is for training phase
        params: it's a python dictionary with the following keys:
            'img_size': a tuple containing width x height
            'channels': an integer representing the image's depth
            'shuffle': set it True if you wanna shuffle the dataset
            'repeat': set it True if you wanna repeat the dataset
            'threads': integer represeting the number of threads to processing the images' load
            'batch_size': an integer representing the batch size
        scalar_feat: if you're also loading scalar features with the images, you
        should use this parameter
        root_folder: 
        params_aug: if you don't wanna have an augmentation, set it as False. Otherwise, if let it None
        this will carry out the augmentation using the default parameters. If you'd like to set your own 
        augmentation parameters, you need to set the dictionary parameters as explained in get_aug_tf above.
    
    Output:
        inputs: a python dictionary containing get_next iterators for the image and labels, and the 
                make_initializable_iterator
        
        dataset: the tf.data.Dataset configured for the given data
'''
def get_dataset_tf(paths, labels, is_train, params, scalar_feat=None, root_folder=None, params_aug=None, verbose=True):
    
    if (scalar_feat is not None):     
        get_aug = lambda x, s, y: get_aug_tf (x, y, s, params_aug)    
        get_img = lambda x, s, y: load_img_as_tensor (x, y, params['img_size'], params['channels'], s, root_folder)
    else:
        get_aug = lambda x, y: get_aug_tf (x, y, None, params_aug)    
        get_img = lambda x, y: load_img_as_tensor (x, y, params['img_size'], params['channels'], root_folder)
    
    if (verbose):   
        if (scalar_feat is not None):
            print ("\n******************\nLoading", len(paths), " images and", labels.shape, "scalar features", "With", labels.shape, " labels\n********************\n")        
        else:
            print ("\n******************\nLoading", len(paths), " images", "With", labels.shape, " labels\n********************\n")        

    
    if (is_train):
        if (scalar_feat is not None):
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(scalar_feat), tf.constant(labels)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
                    
        dataset = dataset.shuffle(len(paths))            
        dataset = dataset.map(get_img, num_parallel_calls=params['threads'])
        
        if (params_aug != False):
            dataset = dataset.map(get_aug, num_parallel_calls=params['threads'])
        
        if (params['repeat']):
            dataset = dataset.repeat()
            
        dataset = dataset.batch(params['batch_size'])
        dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
        
    else:
        if (scalar_feat is not None):
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(scalar_feat), tf.constant(labels)))
        else:            
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
            
        dataset = dataset.map(get_img, num_parallel_calls=params['threads'])
        
        if (params['repeat']):
            dataset = dataset.repeat()
            
        dataset = dataset.batch(params['batch_size'])
        dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    
    if (scalar_feat is not None):
        images, scalar_feat, labels = iterator.get_next()
        inputs = {'images': images, 'scalar_feat': scalar_feat, 'labels': labels, 'iterator_init_op': iterator_init_op}
    else:
        images, labels = iterator.get_next()
        inputs = {'images': images, 'scalar_feat': None, 'labels': labels, 'iterator_init_op': iterator_init_op}
    
    return inputs, dataset

'''
    This function creates a folder tree to populate files in it
    
    Input:
        path: the root folder path
        folders: a list of strings representing the name of the folders will be created inside the path folder
        train_test_val: if you wann create TRAIN, TEST and VAL folders
'''
def create_dirs (path, folders=['A', 'B'], train_test_val=False):        
    
    # Checking if the folder already exists
    if (not os.path.isdir(path)):
        os.mkdir(path)
        
    if (train_test_val):
        if (not os.path.isdir(path + '/' + 'TEST')):
            os.mkdir(path + '/' + 'TEST')
        if (not os.path.isdir(path + '/' + 'TRAIN')):
            os.mkdir(path + '/' + 'TRAIN')                
        if (not os.path.isdir(path + '/' + 'VAL')):
            os.mkdir(path + '/' + 'VAL')
        
    for folder in folders:          
        if (train_test_val):
            if (not os.path.isdir(path + '/TRAIN/' + folder)):
                os.mkdir(path + '/TRAIN/' + folder)         
            if (not os.path.isdir(path + '/TEST/' + folder)):                                                   
                os.mkdir(path + '/TEST/' + folder)
            if (not os.path.isdir(path + '/VAL/' + folder)):
                os.mkdir(path + '/VAL/' + folder)
        else:               
            if (not os.path.isdir(path + '/' + folder)):
                os.mkdir(path + '/' + folder)


'''
    It gets as input a path tree without train, test and validation sets and returns a new folder tree with all sets.
    It's easier to explain with using an example (lol).
        Dataset:
            A:
                img...
            B: 
                img...
    It returns:
        Dataset:
            TRAIN:
                A:
                    imgs...
                B:
                    imgs...
            TEST:
                A:
                    imgs...
                B:
                    imgs...
            VAL:
                A:
                    imgs...
                B:
                    imgs...
    
    Input:
        path_in: the root folder that you wanna split in the train, test and val sets
        path_out: the root folder that will receive the new tree organization
        tr: a float meaning the % of images for the training set
        te: a float meaning the % of images for the test set
        tv: a float meaning the % of images for the validation set
        shuf: set it as True if you wanna shuffle the images
        verbose: set it as True to print information on the screen
    Outpur:
        The new folder tree with all images splited into train, test and val
    

'''
def split_folders_train_test_val (path_in, path_out, scalar_feat_ext=None, img_ext="jpg", tr=0.8, te=0.1, tv=0.1, shuf=True, verbose=False):
        
    if (tr+te+tv != 1.0):
        print ('tr, te and tv must sum up 1.0')
        raise ValueError
        
    folders = [nf for nf in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, nf))]
    
    create_dirs (path_out, folders, True)   
    
    for lab in folders:            
        path_imgs = glob.glob(path_in + '/' + lab + '/*.'+img_ext)        
        
        if shuf:
            shuffle(path_imgs)
        
        N = len(path_imgs)
        n_test = int(round(te*N))
        n_val = int(round(tv*N))
        n_train = N - n_test - n_val        
        
        if (verbose):
            print ('Working on ', lab)
            print ('Total: ', N, ' | Train: ', n_train, ' | Test: ', n_test, ' | Val: ', n_val, '\n')
        
        path_test = path_imgs[0:n_test]
        path_val = path_imgs[n_test:(n_test+n_val)]
        path_train = path_imgs[(n_test+n_val):(n_test+n_val+n_train)]
        
        
        if (scalar_feat_ext is None):
            for p in path_test:
                os.system('cp ' + p + ' ' + path_out + '/TEST/' + lab ) 
                
            for p in path_train:
                os.system('cp ' + p + ' ' + path_out + '/TRAIN/' + lab)
                
            for p in path_val:
                os.system('cp ' + p + ' ' + path_out + '/VAL/' + lab )
        else:
            for p in path_test:
                os.system('cp ' + p + ' ' + path_out + '/TEST/' + lab ) 
                os.system('cp ' + p.split('.')[0] + '_feat.' + scalar_feat_ext + ' ' + path_out + '/TEST/' + lab ) 
                
                
            for p in path_train:
                os.system('cp ' + p + ' ' + path_out + '/TRAIN/' + lab)
                os.system('cp ' + p.split('.')[0] + '_feat.' + scalar_feat_ext + ' ' + path_out + '/TRAIN/' + lab)
                
            for p in path_val:
                os.system('cp ' + p + ' ' + path_out + '/VAL/' + lab )
                os.system('cp ' + p.split('.')[0] + '_feat.' + scalar_feat_ext + ' ' + path_out + '/VAL/' + lab )
            
            
            
            
            
            
            
            