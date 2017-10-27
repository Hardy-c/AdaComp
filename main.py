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


    



