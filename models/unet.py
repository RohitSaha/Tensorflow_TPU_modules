'''
Convolutional Networks for Biomedical Image Segmentation
arxiv: https://arxiv.org/abs/1505.04597
'''

import tensorflow as tf

from central_reservoir.utils.layers import linear
from central_reservoir.utils.layers import conv_batchnorm_relu
from central_reservoir.utils.layers import upconv_2D
from central_reservoir.utils.layers import maxpool
from central_reservoir.utils.layers import avgpool

VALID_ENDPOINTS = (
    'Encode_1',
    'Encode_2',
    'Encode_3',
    'Encode_4',
    'Encode_5',
    'Decode_1',
    'Decode_2',
    'Decode_3',
    'Decode_4',
)


def pad_height(tensor):
    padding = tf.constant(
        [[0, 0], [1, 0], [0, 0], [0, 0]])
    padded_tensor = tf.pad(
        tensor,
        padding,
        'SYMMETRIC')
    return padded_tensor

def pad_width(tensor):
    padding = tf.constant(
        [[0, 0], [0, 0], [1, 0], [0, 0]])
    padded_tensor = tf.pad(
        tensor,
        padding,
        'SYMMETRIC')
    return padded_tensor

def no_pad(tensor):
    return tensor

def pad_features(tensor):
    dynamic_shape = tf.shape(tensor)
    d_height, d_width = dynamic_shape[1], dynamic_shape[2]
    tensor = tf.cond(
        tf.equal(
            tf.mod(
                d_height,
                2),
            1),
        lambda: pad_height(tensor),
        lambda: no_pad(tensor))
    tensor = tf.cond(
        tf.equal(
            tf.mod(
                d_width,
                2),
            1),
        lambda: pad_width(tensor),
        lambda: no_pad(tensor))

    return tensor

def merge_features(tensor_1, tensor_2):
    # tensor_2 > tensor_1
    #shape_1 = tf.shape(tensor_1)
    shape_2 = tf.shape(tensor_2)
    #h1, w1 = shape_1[1], shape_1[2]
    h2, w2 = shape_2[1], shape_2[2]
    #h_diff = tf.subtract(h2, h1)
    #w_diff = tf.subtract(w2, w1)
    new_tensor = tf.image.resize_images(tensor_1, [h2, w2]) 
    #h_padding = tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]])
    #w_padding = tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]])
    #h_pad = h_padding * h_diff
    #w_pad = w_padding * w_diff
    #c_val = h_pad + w_pad

    #new_tensor = tf.pad(tensor_1, c_val, 'SYMMETRIC')

    merged_tensor = tf.concat([new_tensor, tensor_2], axis=-1)

    return merged_tensor


def build_unet(final_endpoint='Decode_4',
                use_batch_norm=False,
                use_cross_replica_batch_norm=False,
                num_cores=8,
                num_classes=92,
                model_scope_name='unet_1'):

    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError('Unknown final endpoint %s' %final_endpoint)

    def model(inputs, is_training):

        net = inputs
        layer_outputs = {}

        inputs = pad_features(inputs)

        print('Input: {}'.format(net.get_shape().as_list()))

        with tf.variable_scope(model_scope_name):
            # Encode_1
            end_point = 'Encode_1'
            with tf.variable_scope(end_point):
                # 3x3 Conv, padding='same'
                conv2d_1a = conv_batchnorm_relu(net, 'Conv2d_1a', num_classes,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_1a

                # 3x3 Conv, padding='same'
                conv2d_1b = conv_batchnorm_relu(conv2d_1a, 'Conv2d_1b', num_classes,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_1b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_1b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_1b
                conv2d_1b = pad_features(conv2d_1b)

                # 2x2 MaxPool
                maxpool_1a = maxpool(conv2d_1b, 'MaxPool_1a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_1a

            if final_endpoint == end_point: return maxpool_1a, layer_outputs

            # Encode_2
            end_point = 'Encode_2'
            with tf.variable_scope(end_point):
                # 3x3 Conv, padding='same'
                conv2d_2a = conv_batchnorm_relu(maxpool_1a, 'Conv2d_2a', num_classes*2,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_2a

                # 3x3 Conv, padding='same'
                conv2d_2b = conv_batchnorm_relu(conv2d_2a, 'Conv2d_2b', num_classes*2,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_2b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_2b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_2b
                conv2d_2b = pad_features(conv2d_2b)

                # 2x2 MaxPool
                maxpool_2a = maxpool(conv2d_2b, 'MaxPool_2a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_2a

            if final_endpoint == end_point: return maxpool_2a, layer_outputs

            # Encode_3
            end_point = 'Encode_3'
            with tf.variable_scope(end_point):
                # 3x3 Conv, padding='same'
                conv2d_3a = conv_batchnorm_relu(maxpool_2a, 'Conv2d_3a', num_classes*4,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_3a

                # 3x3 Conv, padding='same'
                conv2d_3b = conv_batchnorm_relu(conv2d_3a, 'Conv2d_3b', num_classes*4,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_3b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_3b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_3b
                conv2d_3b = pad_features(conv2d_3b)

                # 2x2 MaxPool
                maxpool_3a = maxpool(conv2d_3b, 'MaxPool_3a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_3a

            if final_endpoint == end_point: return maxpool_3a, layer_outputs


            # Encode_4
            end_point = 'Encode_4'
            with tf.variable_scope(end_point):
                # 3x3 Conv, padding='same'
                conv2d_4a = conv_batchnorm_relu(maxpool_3a, 'Conv2d_4a', num_classes*8,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_4a

                # 3x3 Conv, padding='same'
                conv2d_4b = conv_batchnorm_relu(conv2d_4a, 'Conv2d_4b', num_classes*8,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_4b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_4b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_4b
                conv2d_4b = pad_features(conv2d_4b)

                # 2x2 MaxPool
                maxpool_4a = maxpool(conv2d_4b, 'MaxPool_4a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_4a

            if final_endpoint == end_point: return maxpool_4a, layer_outputs


            # Encode_5
            end_point = 'Encode_5'
            with tf.variable_scope(end_point):
                # 3x3 Conv, padding='same'
                conv2d_5a = conv_batchnorm_relu(maxpool_4a, 'Conv2d_5a', num_classes*16,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_5a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_5a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_5a

                # 3x3 Conv, padding='same'
                conv2d_5b = conv_batchnorm_relu(conv2d_5a, 'Conv2d_5b', num_classes*16,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_5b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_5b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_5b

            layer_outputs[end_point] = conv2d_5b
            if final_endpoint == end_point: return conv2d_5b, layer_outputs


            # Decode_1
            end_point = 'Decode_1'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_1a = upconv_2D(conv2d_5b, 'UpConv2d_1a', num_classes*8,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_1a

                # Merge
                #merge_1a = tf.concat(
                #    [conv2d_4b, upconv2d_1a],
                #    axis=-1,
                #    name='merge_1a')
                merge_1a = merge_features(conv2d_4b, upconv2d_1a)
                get_shape = merge_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_1a

                # 3x3 Conv, padding='same'
                conv2d_d_1a = conv_batchnorm_relu(merge_1a, 'Conv2d_d_1a', num_classes*8,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_1a

                # 3x3 Conv, padding='same'
                conv2d_d_1b = conv_batchnorm_relu(conv2d_d_1a, 'Conv2d_d_1b', num_classes*8,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_1b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_1b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_1b

            layer_outputs[end_point] = conv2d_d_1b
            if final_endpoint == end_point: return conv2d_d_1b, layer_outputs

     
            # Decode_2
            end_point = 'Decode_2'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_2a = upconv_2D(conv2d_d_1b, 'UpConv2d_2a', num_classes*4,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_2a

                # Merge
                #merge_2a = tf.concat(
                #    [conv2d_3b, upconv2d_2a],
                #    axis=-1,
                #    name='merge_2a')
                merge_2a = merge_features(conv2d_3b, upconv2d_2a)
                get_shape = merge_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_2a

                # 3x3 Conv, padding='same'
                conv2d_d_2a = conv_batchnorm_relu(merge_2a, 'Conv2d_d_2a', num_classes*4,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_2a

                # 3x3 Conv, padding='same'
                conv2d_d_2b = conv_batchnorm_relu(conv2d_d_2a, 'Conv2d_d_2b', num_classes*4,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_2b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_2b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_2b

            layer_outputs[end_point] = conv2d_d_2b
            if final_endpoint == end_point: return conv2d_d_2b, layer_outputs


            # Decode_3
            end_point = 'Decode_3'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_3a = upconv_2D(conv2d_d_2b, 'UpConv2d_3a', num_classes*2,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_3a

                # Merge
                #merge_3a = tf.concat(
                #    [conv2d_2b, upconv2d_3a],
                #    axis=-1,
                #    name='merge_3a')
                merge_3a = merge_features(conv2d_2b, upconv2d_3a)
                get_shape = merge_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_3a

                # 3x3 Conv, padding='same'
                conv2d_d_3a = conv_batchnorm_relu(merge_3a, 'Conv2d_d_3a', num_classes*2,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_3a

                # 3x3 Conv, padding='same'
                conv2d_d_3b = conv_batchnorm_relu(conv2d_d_3a, 'Conv2d_d_3b', num_classes*2,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_3b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_3b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_3b

            layer_outputs[end_point] = conv2d_d_3b
            if final_endpoint == end_point: return conv2d_d_3b, layer_outputs


            # Decode_4
            end_point = 'Decode_4'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_4a = upconv_2D(conv2d_d_3b, 'UpConv2d_4a', num_classes,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_4a

                # Merge
                #merge_4a = tf.concat(
                #    [conv2d_1b, upconv2d_4a],
                #    axis=-1,
                #    name='merge_4a')
                merge_4a = merge_features(conv2d_1b, upconv2d_4a)
                get_shape = merge_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_4a

                # 3x3 Conv, padding='same'
                conv2d_d_4a = conv_batchnorm_relu(merge_4a, 'Conv2d_d_4a', num_classes,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_4a

                # 3x3 Conv, padding='same'
                conv2d_d_4b = conv_batchnorm_relu(conv2d_d_4a, 'Conv2d_d_4b', num_classes,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_d_4b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_4b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_4b

                # 1x1 Conv, padding='same'
                conv2d_d_4c = conv_batchnorm_relu(conv2d_d_4b, 'Conv2d_d_4c', 93,
                    kernel_size=1, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                    activation=None)
                get_shape = conv2d_d_4c.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_4c'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_4c

            layer_outputs[end_point] = conv2d_d_4c
            if final_endpoint == end_point: return conv2d_d_4c, layer_outputs

    return model


def UNET(final_endpoint='Decode_4', use_batch_norm=False,
        use_cross_replica_batch_norm=False, num_cores=8, num_classes=92,
        model_scope_name='unet_1'):

    return build_unet(
        final_endpoint=final_endpoint,
        use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm,
        num_cores=num_cores,
        num_classes=num_classes,
        model_scope_name=model_scope_name)
