import tensorflow as tf

def DarknetConv(input, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = tf.pad(x, [[1,0],[1,0]], mode="CONSTANT")
        padding = 'vaild'
    x = tf.compat.v1.layers.conv2d(inputs=input, filters=filter, kernel_size=size, strides=strides, padding=padding, kernel_regularizer=tf.)


