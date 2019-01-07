#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:58:59 2018

@author: labcin
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import os
from tqdm import trange
from ..evaluate.evaluate import eval_batch

def _train_batch (sess, train_model_dict, num_steps_train, 
                train_writer, steps_to_save_summary):
    
    # Getting the operations from the dict
    loss = train_model_dict['loss']
    train_op = train_model_dict['train_op']
    iterator_init_op = train_model_dict['iterator_init_op']
    
    metrics_init_op = train_model_dict['metrics_init_op']
    update_metrics = train_model_dict['update_metrics']
    metrics = train_model_dict['metrics']
    
    summary_op = train_model_dict['summary_op']
    global_step = tf.train.get_global_step()
    
    # For every batch we need to restart the dataset iterator in order to get the new batch of data
    # In addition, we need to initialize the metrics as well
    sess.run(iterator_init_op)
    sess.run(metrics_init_op)

    # Writing the progress bar using tqdm
    t = trange(num_steps_train)
    
    for i in t:
        
        # Checking if we need to use the summaty to generate new data in tensorboard
        if (i % steps_to_save_summary == 0):
            
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                summary_op, global_step])
            # Write summaries for tensorboard
            train_writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
            
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))


    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    print ("- Train metrics: " + metrics_string)


def train_model (train_model_dict, eval_model_dict,
            params, sess=None, save_dir="model", load_weights_from=None,
            steps_to_save_summary=5):

    print ("Starting the training phase...")
    
    # Keeping the best and the last checkpoint
    best_model = tf.train.Saver(max_to_keep=1)
    last_model = tf.train.Saver(max_to_keep=1)
    init_epoch = 0

    # Checking the session
    if (sess is None):
        sess = tf.Session()

    # Initialing the writers for tensorboard
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train_summaries'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(save_dir, 'eval_summaries'), sess.graph)

    # Initializing the model variables
    sess.run(train_model_dict['variable_init_op'])

    # Checking if we need to restore the model. If yes, do it:
    if (train_model_dict['slim_restore_fn'] is not None):
        slim_restore_fn = train_model_dict['slim_restore_fn']
        slim_restore_fn(sess)

    # Checking if it is a reload
    if (load_weights_from is not None):
        print ("Reloading the model from {}".format(load_weights_from))

        if (os.path.isdir(load_weights_from)):
            reload_from = tf.train.latest_checkpoint(load_weights_from)
            init_epoch = int(load_weights_from.split('-')[-1])
        last_model.save(sess, reload_from)

    best_eval_acc = 0.0

    # Computing the number of steps needed to process one batch of the train dataset
    num_steps_train = (params['train_size'] + params['batch_size'] -1) // params['batch_size']

    for epoch in range(init_epoch, init_epoch + params['num_epochs']):

        print ("Starting epoch {} of {}".format(epoch, params['num_epochs']))


        _train_batch (sess, train_model_dict, num_steps_train, train_writer, steps_to_save_summary)

        # Save weights last weight for each epochTrue
        last_save_path = os.path.join(save_dir, 'last_weights', 'after-epoch')
        last_model.save(sess, last_save_path, global_step=epoch + 1)


        # Evaluating the model
        metrics, _ = eval_batch (eval_model_dict, sess, params, eval_writer)

        # Gettinh the best evaluation model to save it
        eval_acc = metrics['accuracy']
        if (eval_acc >= best_eval_acc):

            # Store new best accuracy
            best_eval_acc = eval_acc

            # Save weights
            best_save_path = os.path.join(save_dir, 'best_weights', 'after-epoch')
            best_save_path = best_model.save(sess, best_save_path, global_step=epoch + 1)
            print ("- Found new best accuracy ({}). Saving the model in {}".format(best_eval_acc, best_save_path))

    sess.close()
        
        

        
        
        
        
        
        
        
        
        
        
        
        