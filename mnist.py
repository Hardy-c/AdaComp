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

import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf


# mnist.py is inspired by cifar10_input.py from the cifar10 tutorial of Tensorflow (see https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', './', """Path to the MNIST data directory.""")

import os
import tensorflow as tf


IMAGE_SIZE = 28

# Global constants describing the Mnist data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_mnist(filename_queue_images,filename_queue_labels):
      class MNISTRecord(object):
            pass
      result = MNISTRecord()
      label_bytes = 1
      result.height = 28
      result.width = 28
      result.depth = 1
      image_bytes = result.height * result.width * result.depth

      reader_images = tf.FixedLengthRecordReader(record_bytes=image_bytes)
      reader_labels = tf.FixedLengthRecordReader(record_bytes=label_bytes)
      
      result.key_images, value_images = reader_images.read(filename_queue_images)
      result.key_labels, value_labels = reader_labels.read(filename_queue_labels)
      
      
      # Convert from a string to a vector of uint8 that is record_bytes long.
      record_bytes_images = tf.decode_raw(value_images, tf.uint8)
      record_bytes_labels = tf.decode_raw(value_labels, tf.uint8)
    
      # The label is converted from uint8->int32.
      result.label = tf.reshape(tf.cast(record_bytes_labels, tf.int32),[1])


      # We reshape the image from [depth * height * width] to [depth, height, width].
      depth_major = tf.reshape(record_bytes_images,
                               [result.depth, result.height, result.width])
      # Convert from [depth, height, width] to [height, width, depth].
      result.uint8image = tf.transpose(depth_major, [1, 2, 0])
      
      return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
      # Create a queue that shuffles the examples, and then
      # read 'batch_size' images + labels from the example queue.
      num_preprocess_threads = 1
      if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                  [image, label],
                  batch_size=batch_size,
                  num_threads=num_preprocess_threads,
                  capacity=min_queue_examples + 3 * batch_size,
                  min_after_dequeue=min_queue_examples)
      else: # Test dataset
            images, label_batch = tf.train.batch(
                  [image, label],
                  batch_size=batch_size,
                  num_threads=num_preprocess_threads,
                  capacity=10000)
        
      

      return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size, start,stop):
    
    filenames_images = [os.path.join(data_dir, 'training_set_images/train-images-%05d.idx3-ubyte' % i)
                 for i in range(start, stop)]
    filenames_labels = [os.path.join(data_dir, 'training_set_labels/train-labels-%05d.idx1-ubyte' % i)
                 for i in range(start, stop)]
    for f in filenames_images:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
        
    # Create a queue that produces the filenames to read.
    filename_queue_images = tf.train.string_input_producer(filenames_images,shuffle=False)
    filename_queue_labels = tf.train.string_input_producer(filenames_labels,shuffle=False)
    
    # Read examples from files in the filename queue.
    read_input = read_mnist(filename_queue_images,filename_queue_labels)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    float_image = tf.image.per_image_standardization(reshaped_image)
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.2/100
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d MNIST images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(data_dir, batch_size):
    
    filenames_images = [os.path.join(data_dir, 'test_set_images/test-images-%05d.idx3-ubyte' % i) for i in range(200)]
    filenames_labels = [os.path.join(data_dir, 'test_set_labels/test-labels-%05d.idx1-ubyte' % i) for i in range(200)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    
    for f in filenames_images:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue_images = tf.train.string_input_producer(filenames_images,shuffle=False)
    filename_queue_labels = tf.train.string_input_producer(filenames_labels,shuffle=False)
    
    # Read examples from files in the filename queue.
    read_input = read_mnist(filename_queue_images,filename_queue_labels)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    float_image = tf.image.per_image_standardization(reshaped_image)

        
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)




def get_local_data(current_worker_id=0,total_number_of_worker=1):
  examples_per_worker = 1200/total_number_of_worker
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return distorted_inputs(data_dir=FLAGS.data_dir,batch_size=FLAGS.batch_size,start=int((current_worker_id-1)*examples_per_worker),stop=int((current_worker_id)*examples_per_worker))


def get_test_set():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
