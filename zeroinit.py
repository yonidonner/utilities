import numpy

import tensorflow as tf

def match_to_size(from_layer, to_layer, ignore_axes=1):
    from_shp = from_layer.get_shape().as_list()
    to_shp = to_layer.get_shape().as_list()
    if len(from_shp) != len(to_shp):
        raise ValueError, 'source and target have different ranks'
    d = len(from_shp)
    if from_shp[0] != to_shp[0]:
        raise ValueError, 'source and target have different batch size'
    axes = [i for i in range(1, d-ignore_axes) if from_shp[i] != to_shp[i]]
    if len(axes) == 0:
        return from_layer
    begin = [0] * d
    size = [-1] * d
    for i in axes:
        if from_shp[i] < to_shp[i]:
            raise ValueError, 'source smaller than target, padding not supported'
        dif = from_shp[i] - to_shp[i]
        if (dif % 2) != 0:
            raise ValueError, 'difference %d not even, shapes %s and %s'%(
                dif, from_shp, to_shp)
        begin[i] = dif // 2
        size[i] = to_shp[i]
    return tf.slice(from_layer, begin, size)

def run_unet(run_forward, run_down, run_up, block_params, layer):
    lateral = []
    for (i, my_params) in enumerate(block_params):
        name = 'down%d'%(i+1,) 
        (pre_params, down_params, up_params, post_params) = my_params
        if i > 0:
            layer = run_down(name, layer, down_params)
        layer = run_forward(name, layer, pre_params)
        lateral.append(layer)
    lateral.reverse()
    for (i, my_params) in enumerate(reversed(block_params)):
        name = 'up%d'%(i+1,)
        (pre_params, down_params, up_params, post_params) = my_params
        if i > 0:
            layer = run_up(name, layer, up_params)
            lat = match_to_size(lateral[i], layer)
            layer = tf.concat([lat, layer], len(lat.shape) - 1)
        layer = run_forward(name, layer, post_params)
    return layer

def run_down_avgpool(name, layer, padding='SAME'):
    return tf.nn.avg_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], padding=padding)

def run_up_resize(name, layer):
    size = [layer.shape[1] * 2, layer.shape[2] * 2]
    return tf.image.resize_images(layer, size)

def add_bias(name, layer):
    bias = tf.get_variable(
        name, [layer.shape[-1]], layer.dtype, tf.zeros_initializer())
    tf.add_to_collection(tf.GraphKeys.BIASES, bias)
    return layer + bias

def add_multiplier(name, layer):
    mult = tf.get_variable(
        name, [layer.shape[-1]], layer.dtype, tf.ones_initializer())
    tf.add_to_collection('multipliers', mult)
    return layer * mult

def make_initializer(scale):
    if scale > 0:
        return tf.random_normal_initializer(mean=0.0, stddev=numpy.sqrt(scale))
    return tf.zeros_initializer()

def make_strides(stride, n=4):
    if numpy.isscalar(stride):
        return [stride] * n
    return stride

def conv2d(
        name, layer, n_out=None, k=3, stride=1, padding='SAME', rate=1, scale=1):
    n_in = int(layer.shape[3])
    if n_out is None:
        n_out = n_in
    fan_in = k * k * n_in
    W = tf.get_variable(
        name, [k, k, n_in, n_out], dtype=layer.dtype,
        initializer=make_initializer(scale / float(fan_in)))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    return tf.nn.conv2d(
        layer, W, make_strides(stride), padding=padding,
        dilations=make_strides(rate))

def zeroinit_orig_block(name, layer, L, padding='SAME'):
    shortcut = layer
    with tf.variable_scope(name):
        layer = add_bias('bias1', layer)
        layer = conv2d('conv1', layer, padding=padding, scale=2.0/L)
        layer = add_bias('bias3', tf.nn.relu(add_bias('bias2', layer)))
        layer = conv2d('conv2', layer, padding=padding, scale=0)
        layer = add_multiplier('mult', layer)
        if padding != 'SAME':
            shortcut = match_to_size(shortcut, layer)
        tf.add_to_collection('residuals', layer)
        layer = tf.nn.relu(add_bias('bias4', layer + shortcut))
    return layer

def residual_block(name, layer, resfunc, padding='SAME'):
    with tf.variable_scope(name):
        residual = resfunc(layer, padding=padding)
    tf.add_to_collection('residuals', residual)
    if padding != 'SAME':
        layer = match_to_size(layer, residual)
    return layer + residual

def zeroinit_preactivation_resfunc(layer, L, padding='SAME'):
    layer = add_bias('bias2', tf.nn.relu(add_bias('bias1', layer)))
    layer = conv2d('conv1', layer, padding=padding, scale=2.0/L)
    layer = add_bias('bias4', tf.nn.relu(add_bias('bias3', layer)))
    layer = conv2d('conv2', layer, padding=padding, scale=0)
    layer = add_multiplier('mult', layer)
    return layer

def zeroinit_pyramid_bottleneck_block(
        name, layer, growth, bottleneck, L, padding='SAME'):
    out_size = int(layer.shape[3])+growth
    with tf.variable_scope(name):
        residual = add_bias('bias1', layer)
        if (bottleneck is None) or (bottleneck >= out_size):
            residual = conv2d('conv1', residual, out_size, 3, scale=2.0/L)
            residual = add_bias('bias3', tf.nn.relu(add_bias('bias2', residual)))
            residual = conv2d('conv2', residual, out_size, 3, scale=0)
        else:
            sc = numpy.power(L, -0.5)
            residual = conv2d('conv1', residual, bottleneck, 1, scale=2*sc)
            residual = add_bias('bias3', tf.nn.relu(add_bias('bias2', residual)))
            residual = conv2d('conv2', residual, bottleneck, 3, scale=2*sc)
            residual = add_bias('bias5', tf.nn.relu(add_bias('bias4', residual)))
            residual = conv2d('conv3', residual, out_size, 1, scale=0)
        residual = add_multiplier('mult', residual)
        tf.add_to_collection('residuals', residual)
        if padding != 'SAME':
            layer = match_to_size(layer, residual)
        if growth > 0:
            layer = tf.concat(
                [layer, conv2d('conv_shortcut', layer, growth, 1, scale=1)], 3)
        elif growth < 0:
            layer = conv2d('conv_shortcut', layer, out_size, 1, scale=1)
    return layer + residual
            
def dense_compress(
        name, layer, out_size, k, growth):
    with tf.variable_scope(name):
        layer1 = layer
        for i in range(k):
            layer2 = conv2d('conv%d'%(i+1,), layer1, growth, k=1, scale=2)
            layer2 = tf.nn.relu(add_bias('bias%d_1'%(i+1,), layer2))
            layer2 = add_mult('mult%d'%(i+1,), layer2)
            layer2 = add_bias('bias%d_2'%(i+1,), layer2)
            layer1 = tf.concat([layer1, layer2], 3)
        layer = conv2d('conv_compress', layer1, out_size, k=1, scale=1)
    return layer
