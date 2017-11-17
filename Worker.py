import tensorflow as tf
import Tf_op as df
import Communication as com
import socket as sck
import Model_DNN as mdnn
import mnist



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a mini-batch.""")

tf.app.flags.DEFINE_integer('compression_rate', 0.01,
                            """Compression rate of worker updates.""")



# Worker routine
def worker(D,graph=None):
  """ Build Tensorflow graph and run iterations """

  if graph ==None:
    graph = tf.Graph()
  # Build Tensorflow graph which computes gradients of the model with one mini-batch of examples
  with graph.as_default():
          
    # Get input and labels for learning from D
    inputs, labels = D
    logits = mdnn.CNN_model(inputs,graph)
    
    # Calculate loss.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
   
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    grads = optimizer.compute_gradients(loss)
    with tf.variable_scope("",reuse=True):
      grads_var = {var.op.name:tf.Variable(tf.zeros(var.get_shape()),trainable=False,name=var.op.name+"_grad",collections=["W_grad"]) for _,var in grads}
    train_op = [grads_var[var.op.name].assign(grad) for grad,var in grads]
    
    # Build an initialization operation.
    init = tf.global_variables_initializer()

    
    # Tensorflow op to update parameters from PS
    get_W = df.get_w(graph,"W_global")


    with tf.Session() as sess:
      #Initialize the TF variables
      sess.run([init])
      tf.train.start_queue_runners(sess=sess)
      iteration = 0
      s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
      s.connect((FLAGS.ip_PS, FLAGS.port))
      
      while iteration < FLAGS.iter_max:
        #Get the parameters from the PS
        com.send_msg(s,"","GET_W")
        cmd,data= com.recv_msg(s)
        iteration,W= com.decode_variables(data)
        s.close()
        
        #Update the parameters
        sess.run(get_W,{key+"_delta:0":value for key,value in W.items()})
        
        #Compute gradients stored in Tensorflow variables
        inp,log,lab,loss_values,_ =sess.run([inputs,logits,labels,loss,train_op])

        print "Loss",loss_values
        
        #Encode the update with the local timer (iteration)
        update = com.encode_variables(sess,"W_grad",iteration,compression=FLAGS.compression_rate)
        
        #Push the update to PS
        s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
        s.connect((FLAGS.ip_PS, FLAGS.port))
        
        com.send_msg(s,update,"PUSH")
      print "Worker",FLAGS.id_worker," is closed"
      
