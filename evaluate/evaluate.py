#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:17:25 2018

@author: labcin
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import os
    
def eval_batch (eval_model_dict, sess, params, eval_writer=None):

    update_metrics = eval_model_dict['update_metrics']
    eval_metrics = eval_model_dict['metrics']
    iterator_init_op = eval_model_dict['iterator_init_op']
    metrics_init_op = eval_model_dict['metrics_init_op']
    predictions = eval_model_dict["predictions"]
    global_step = tf.train.get_global_step()

    # For every batch we need to restart the dataset iterator in order to get the new batch of data
    # In addition, we need to initialize the metrics as well
    sess.run(iterator_init_op)
    sess.run(metrics_init_op)

    num_steps_eval = (params['eval_size'] + params['batch_size'] -1) // params['batch_size']
    pred_list = list()
    # compute metrics over the dataset
    for _ in range(num_steps_eval):
        p, _ = sess.run([predictions, update_metrics])
        pred_list += p.tolist()

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    print ("- Eval metrics: " + metrics_string)

    # # Add summaries manually to writer at global_step_val
    if (eval_writer is not None):
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            eval_writer.add_summary(summ, global_step_val)


    return metrics_val, pred_list


def testing_model (test_model_dict, params, weights_to_load, model_save_dir="model"):

    # Initialize tf.Saver
    saver = tf.train.Saver()

    # Just adjusting the test_size
    if ('test_size' in params.keys()):
        params['eval_size'] = params['test_size']

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(test_model_dict['variable_init_op'])

        # test_writer = tf.summary.FileWriter(os.path.join(model_save_dir, 'test_summaries'), sess.graph)
        
        if os.path.isdir(weights_to_load):
            weights_to_load = tf.train.latest_checkpoint(weights_to_load)
        saver.restore(sess, weights_to_load)

        # Evaluate
        print ("\nResults for the testing dataset:")
        return eval_batch (test_model_dict, sess, params)
        # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        # print ("- Test metrics: " + metrics_string)
