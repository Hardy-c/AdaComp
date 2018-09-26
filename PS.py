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
import Tf_op as df
import socket as sck
import numpy as np
import Model_DNN as mdnn
import Communication as com
# import mnist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('learning_rate', -0.1,
                            """ Learning rate of the gradient descent.""")


# ADACOMP - compute staleness
def count_pred(history,key,loc):
    if history == [] :
        st = [0 for i in loc]
    else:
        st = np.add.reduce([[1 if i in h[key] else 0 for i in loc] for h in history], axis=0)
    return st


# Parameter Server routine
def PS():
    with tf.Graph().as_default() as graph:
      
        # Get input and labels for learning from D
        inputs, labels = tf.placeholder(tf.float32,shape=[None,FLAGS.image_size,FLAGS.image_size,FLAGS.image_depth]), tf.placeholder(tf.float32,shape=[None,FLAGS.nb_classes])
        
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = mdnn.CNN_model(inputs,graph)
        
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        init_glob = tf.variables_initializer(tf.get_collection("W_global"))

        update_op = df.apply_sparse_update(graph,"W_global")

      
        with tf.Session(graph=graph) as sess:
      
            # Initialize the Deep Neural Network
            sess.run([init,init_glob])

            # Configure socket
            tcpsock = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
            tcpsock.setsockopt(sck.SOL_SOCKET, sck.SO_REUSEADDR, 1)
            
            tcpsock.bind(("",FLAGS.port))

            history = []
            iteration = 0
            while iteration <FLAGS.iter_max+FLAGS.nb_workers-1:
                tcpsock.listen(1)
                (wsocket, (ip, port)) = tcpsock.accept()
                cmd,data = com.recv_msg(wsocket)
                if cmd == "GET_W":
                    #Encode parameter
                    parameters = com.encode_variables(sess,"W_global",iteration,compression=1)
                    com.send_msg(wsocket,parameters,"PARAM")
                    wsocket.close()
                elif cmd =="PUSH":
                    old_iter,gradients,indices = com.decode_variables(data)
                    delay = iteration-old_iter

                    # for each trainable variable of the model
                    for k in gradients.keys():
                        #Compute staleness for each parameter
                        staleness = count_pred(history[-delay:],k,indices[k])
                        gradients[k] = [FLAGS.learning_rate*gradients[k][i]/max(1,staleness[i]) for i in range(len(gradients[k]))]
                    
                    # Update paramters
                    feed_dict = {}
                    for k in gradients.keys():
                        feed_dict[k[:-5]+"_delta:0"]=gradients[k]
                        feed_dict[k[:-5]+"_delta_indices:0"]=indices[k]

                    sess.run(update_op,feed_dict)
                    # Add update to history
                    history.append({key:set(indices)  for key,indices in indices.items()})

                    iteration+=1
                    cmd,data = com.recv_msg(wsocket)
                    if cmd == "GET_W":
                        #Encode parameter
                        parameters = com.encode_variables(sess,"W_global",iteration,compression=1)
                        com.send_msg(wsocket,parameters,"PARAM")
                        wsocket.close()
                    else:
                        wsocket.close()
            
            print "PS is closed"
