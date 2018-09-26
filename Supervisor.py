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
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'''

from __future__ import print_function
import tensorflow as tf
import Tf_op as df
import socket as sck
import Model_DNN as mdnn
import Communication as com
import mnist
import time

FLAGS = tf.app.flags.FLAGS


def Supervisor(D,graph=None):
  """ Build Tensorflow graph and run iterations """
  if graph ==None:
    graph = tf.Graph()
  # Build Tensorflow graph
  with graph.as_default():
    
    # Get Test inputs and labels for compute accuracy from D
    inputs, labels = D
          
    # Build a Graph that computes the logits predictions from the model.
    logits = mdnn.CNN_model(inputs,graph)
    
    # Calculate loss and accuracy.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,labels,1),tf.float32))

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    #init_glob = tf.variables_initializer(tf.get_collection("W_global"))

    # Tensorflow op to update parameters from PS
    get_W = df.get_w(graph,"W_global")


    with tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=1)) as sess:
        #Initialize TF variables
        sess.run([init])
        tf.train.start_queue_runners(sess=sess)

        iteration = 0
      
        while iteration <FLAGS.iter_max:      
            s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
            s.connect((FLAGS.ip_PS, FLAGS.port))
            
            #Get parameters from PS
            com.send_msg(s,"","GET_W")

            _,data= com.recv_msg(s)
            iteration,W= com.decode_variables(data)
            s.close()

            #Update parameters
            sess.run(get_W,{key+"_delta:0":value for key,value in W.items()})

            #Compute gradients stored in Tensorflow variables
            score,loss_value =  sess.run([accuracy,loss])
            print("---------------------------------------")
            print("Iteration ",iteration)
            print("Test accuracy :",score*100,"%")
            print("Test Loss :",loss_value)
            print("---------------------------------------")
            # Compute accuracy every minute
            time.sleep(60)
              
    

        
