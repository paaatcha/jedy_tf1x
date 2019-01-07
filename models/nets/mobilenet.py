import tensorflow as tf
import tensorflow.contrib.slim as slim


# n_labels, is_training=True, width_multiplier=1, scope='MobileNet', freeze_convs=False
def mobilenet(dict_data, params):

    n_labels = params['n_labels']
    is_training = params['is_training']
    inputs = dict_data['images']

    if ('width_multiplier' not in params.keys()):
        width_multiplier = 1.0
    else:
        width_multiplier = params['width_multiplier']

    if ('scope' not in params.keys()):
        scope = 'MobileNet'
    else:
        scope = params['scope']

    if ('freeze_convs' not in params.keys()):
        freeze_convs = False
    else:
        freeze_convs = params['freeze_convs']

    def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, sc, downsample=False, freeze_convs=False):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv',
                                                      trainable=not freeze_convs)

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm', trainable=not freeze_convs)
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv',
                                            trainable=not freeze_convs)
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm', trainable=not freeze_convs)
        return bn

    # with tf.variable_scope(scope) as sc:
    end_points_collection = '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        outputs_collections=[end_points_collection]):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            activation_fn=tf.nn.relu,
                            fused=True):
            net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME',
                                     scope='conv_1', trainable=not freeze_convs)
            net = slim.batch_norm(net, scope='conv_1/batch_norm', trainable=not freeze_convs)

            net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3',
                                            freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5',
                                            freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7',
                                            freeze_convs=freeze_convs)

            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11', freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12', freeze_convs=freeze_convs)

            net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13',
                                            freeze_convs=freeze_convs)
            net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14', freeze_convs=freeze_convs)
            net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')

            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

            with tf.variable_scope('block_fc1'):
                net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu,
                                      kernel_initializer=slim.xavier_initializer(),
                                      kernel_regularizer=slim.l2_regularizer(0.0001))

            with tf.variable_scope('block_fc2'):
                net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu,
                                      kernel_initializer=slim.xavier_initializer(),
                                      kernel_regularizer=slim.l2_regularizer(0.0001))

            with tf.variable_scope('block_fc3'):
                net = tf.layers.dense(inputs=net, units=n_labels, activation=tf.nn.relu,
                                      kernel_initializer=slim.xavier_initializer(),
                                      kernel_regularizer=slim.l2_regularizer(0.0001))


            return net

                # return slim.flatten(net)

        # end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        # net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        # end_points['squeeze'] = net
        # logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
        # predictions = slim.softmax(logits, scope='Predictions')

        # end_points['Logits'] = logits
        # end_points['Predictions'] = predictions

    # return logits, end_points


def mobilenet_arg_scope(weight_decay=0.0):
    """Defines the default mobilenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the MobileNet model.
    """
    with slim.arg_scope(
            [slim.convolution2d, slim.separable_convolution2d],
            weights_initializer=slim.initializers.xavier_initializer(),
            biases_initializer=slim.init_ops.zeros_initializer(),
            weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc
