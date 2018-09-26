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
import mnist
from Worker import worker
from PS import PS
from Supervisor import Supervisor



FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('iter_max', 100000,
                            """Number of iterations to process on PS.""")
tf.app.flags.DEFINE_string('type_node', 'Worker',
                            """Worker|PS|Superisor : define the local computation node""")
tf.app.flags.DEFINE_integer('nb_workers', 100,
                            """Number of workers.""")
tf.app.flags.DEFINE_integer('id_worker', 0,
                            """ID of worker""")
tf.app.flags.DEFINE_string('ip_PS', '0.0.0.0',
                            """The ip adresse of PS""")
tf.app.flags.DEFINE_integer('port', 2223,
                            """The port used in PS""")
tf.app.flags.DEFINE_integer('image_size', 28,
                            """The size of image""")
tf.app.flags.DEFINE_integer('image_depth', 1,
                            """The depth of image""")
tf.app.flags.DEFINE_integer('nb_classes', 10,
                            """ Number of classes""")




if FLAGS.type_node == "Worker":
    with tf.Graph().as_default() as graph:
        #Load training data
        training_data = mnist.get_local_data(FLAGS.id_worker,FLAGS.nb_workers)
        #Run model
        worker(training_data,graph)

if FLAGS.type_node == "Supervisor":
    with tf.Graph().as_default() as graph:
        #Load test data
        test_data = mnist.get_test_set()
        #Test model
        Supervisor(test_data,graph)

if FLAGS.type_node == "PS":
    #Update model loop
    PS()


    



