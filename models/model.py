#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This class is used to get your CNN model

---

If you find any bug, please email me =)

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim


def _set_optimizer (optimizer, params):
    '''
        Aux function to set the optimizer for the model
    '''


    # First, checking the learning rate
    if ('learning_rate' not in params.keys()):
        raise Exception('You must set the learning rate for the optimizer')


    if (optimizer == "adam"):

        if ('beta1' not in params.keys()):
            beta1 = 0.9
        if ('beta2' not in params.keys()):
            beta2 = 0.999
        if ('epsilon' not in params.keys()):
            epsilon = 1e-08

        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=beta1,
                                           beta2=beta2, epsilon=epsilon)

    elif (optimizer == "rmsprop"):

        if ('decay' not in params.keys()):
            decay = 0.9
        if ('momentum' not in params.keys()):
            momentum = 0.0
        if ('epsilon' not in params.keys()):
            epsilon = 1e-10

        optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'], decay=decay,
                                              momentum=momentum, epsilon=epsilon)

    elif (optimizer == "adagrad"):

        if ("initial_accumulator_value" not in params.key()):
            initial_accumulator_value = 0.1

        optimizer = tf.train.AdagradOptimizer (learning_rate=params['learning_rate'],
                                               initial_accumulator_value="initial_accumulator_value")

    elif (optimizer == "sgd"):

        optimizer = tf.train.GradientDescentOptimizer (learning_rate=params['learning_rate'])

    else:
        raise Exception("You must set a valid optimizer")

    return optimizer


def _set_loss (loss, labels, logits):
    '''
        Aux function to set the loss for the model
    '''

    if (loss == "sparse_softmax_cross_entropy"):
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    elif (loss == "softmax_cross_entropy"):
        return tf.losses.softmax_cross_entropy(labels=labels, logits=logits)

    else:
        raise Exception("You must set a valid loss function")


def create_model (net_fn, dict_input, is_train, is_reuse,
                 params, optimizer="adam",
                 loss="sparse_softmax_cross_entropy",
                 net_name="net_model", img_summary=False,
                 restore_weigths_from=None):
    '''
        This function creates the CNN model for train and/or evaluate

        :param net_fn:
        :param dict_input:
        :param is_train:
        :param params:
        :param optimizer:
        :param loss:
        :param net_name:
        :param img_summary:
        :param verbose:
        :return:
    '''

    labels = dict_input["labels"]
    labels = tf.cast(labels, tf.int64)


    with tf.variable_scope(net_name, reuse = is_reuse):
        if (is_train and restore_weigths_from is not None):

            logits = net_fn(dict_input, params)

            if os.path.isdir(restore_weigths_from):
                restore_weigths_from = tf.train.latest_checkpoint(restore_weigths_from)

            variables_to_restore = slim.get_model_variables()
            slim_restore_fn = slim.assign_from_checkpoint_fn (restore_weigths_from, variables_to_restore)

        else:
            logits = net_fn(dict_input, params)

        predictions = tf.argmax(logits, 1)
    
    loss_op = _set_loss (loss, labels, logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Setting the optimizer for the trainning phase
    if (is_train):
        optimizer_op = _set_optimizer(optimizer, params)
        global_step = tf.train.get_or_create_global_step()
        
        if (params['batch_norm']):
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer_op.minimize(loss_op, global_step=global_step)
        else:
            train_op = optimizer_op.minimize(loss_op, global_step=global_step)
        
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss_op)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', acc)
    tf.summary.image('train_image', dict_input['images'])


    if (img_summary):
        # Add incorrectly labeled images
        mask = tf.not_equal(labels, predictions)

        # Add a different summary to know how they were misclassified
        for label in range(0, params['n_labels']):
            mask_label = tf.logical_and(mask, tf.equal(predictions, label))
            incorrect_image_label = tf.boolean_mask(dict_input['images'], mask_label)
            tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = dict_input
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions    
    model_spec['loss'] = loss_op
    model_spec['accuracy'] = acc
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if (is_train):
        model_spec['train_op'] = train_op

        if (restore_weigths_from is not None):
            model_spec['slim_restore_fn'] = slim_restore_fn
        else:
            model_spec['slim_restore_fn'] = None

    return model_spec
    
    

