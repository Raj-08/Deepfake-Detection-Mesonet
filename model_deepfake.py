import tensorflow as tf
from ops import conv,batch_normalization,unpool_with_argmax,sigmoid,relu,dropout,fc_layer
is_training = True
import tflearn as tfl
def build_model(input_img):
    
    conv1=conv(input_img,3, 3, 8, 1, 1, biased=False, relu=False, name='conv1')
    bn_conv1=batch_normalization(conv1,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')    
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(bn_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    conv3=conv(pool1,5, 5,8, 1, 1, biased=False, relu=False, name='conv3')
    bn_conv3=batch_normalization(conv3,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3')
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(bn_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    conv5=conv(pool2,5, 5, 16, 1, 1, biased=False, relu=False, name='conv5')
    bn_conv5=batch_normalization(conv5,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5')
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(bn_conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    conv6=conv(pool3,5, 5, 16, 1, 1, biased=False, relu=False, name='conv6')
    bn_conv6=batch_normalization(conv6,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6')
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(bn_conv6, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    
    drop1=dropout(pool4,name='drop1')
    fc1 = tfl.layers.core.fully_connected(drop1, 16, activation=None)
    drop2=dropout(fc1,name='drop2')
    fc2 = tfl.layers.core.fully_connected(drop2, 1, activation=None)
    final=sigmoid(fc2,name='sigmoid')
    
    return final,fc2;
