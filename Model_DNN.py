'''Copyright (c) 2015 – Thomson Licensing, SAS
Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the
disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of Thomson Licensing, or Technicolor, nor the names
of its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def CNN_model(inputs,graph):
    with graph.as_default():

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = tf.reshape(tf.get_variable('weights', shape=[3* 3* 1* 32],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[3,3,1,32])
            
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [32],initializer=tf.constant_initializer(0.1),collections=["W_global"])
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.reshape(tf.get_variable('weights', shape=[3* 3* 32* 64],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[3,3,32,64])
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1),collections=["W_global"])
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            

        # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm2')
        
        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('fc3') as scope:
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            weights = tf.reshape(tf.get_variable('weights', shape=[7*7*64*128],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[7*7*64,128])
            biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1),collections=["W_global"])
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            


        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.reshape(tf.get_variable('weights', [128*FLAGS.nb_classes],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[128,FLAGS.nb_classes])
            biases = tf.get_variable('biases', [FLAGS.nb_classes],initializer=tf.constant_initializer(0.1),collections=["W_global"])
            softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
            

    return softmax_linear
