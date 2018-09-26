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
