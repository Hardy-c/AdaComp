import tensorflow as tf


def apply_sparse_update(graph, collection):
  with graph.as_default():
    assign_op_tab = []
    updates = {var.op.name+"_delta": tf.placeholder(tf.float32,shape=[None],name=var.op.name+"_delta") for var in tf.get_collection_ref(collection)}
    updates_indices = {var.op.name+"_delta_indices": tf.placeholder(tf.int32,shape=[None],name=var.op.name+"_delta_indices") for var in tf.get_collection_ref(collection)}

    for var in tf.get_collection_ref(collection):
      update = updates[var.op.name+"_delta"]
      update_ind = updates_indices[var.op.name+"_delta_indices"]
      assign_op = tf.scatter_add(var,update_ind,update,False,"Updater")
      assign_op_tab.append(assign_op)
  return assign_op_tab


def get_w(graph, collection,suffix=""):
  with graph.as_default():
    assign_op_tab = []
    updates = {var.op.name+"_delta"+suffix: tf.placeholder(tf.float32,shape=[None],name=var.op.name+"_delta"+suffix) for var in tf.get_collection_ref(collection)}
    for var in tf.get_collection_ref(collection):
      update = updates[var.op.name+"_delta"+suffix]
      assign_op = tf.assign(var,tf.reshape(update,var.shape,name="Assign_"+var.op.name))
      assign_op_tab.append(assign_op)
      
  return assign_op_tab
