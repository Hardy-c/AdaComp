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

import socket
import struct
import cPickle as pck
import heapq

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_msg(sock, msg, cmd):
    # Prefix each message with a 4-byte length (network byte order)
    cmd = cmd[0]
    assert(isinstance(cmd,str) and len(cmd)==1)
    msg = struct.pack('>I', len(msg))+struct.pack('>c',cmd)+ msg
    sock.sendall(msg)
    
def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    raw_cmd = recvall(sock,1)
    if not raw_cmd :
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    cmd = struct.unpack('>c',raw_cmd)[0]
    if cmd == 'P':
        cmd = "PUSH"
    elif cmd == 'G':
        cmd = "GET_W"
    # Read the message data
    return cmd,recvall(sock, msglen)


# Float class with inverse ordering
class backwards(float):
    def __lt__(self, other):
        return float.__lt__(abs(self), abs(other))
    def __le__(self, other):
        return float.__le__(abs(self), abs(other))
    def __gt__(self, other):
        return float.__gt__(abs(self), abs(other))
    def __ge__(self, other):
        return float.__ge__(abs(self), abs(other))


def encode_variables(sess,collection,iteration,compression=1):
    updates = {}
    # Compression of variables
    if compression<1 :
        indices_dict = {}
        for var in sess.graph.get_collection_ref(collection):
            value = sess.run(var).flatten()
            nb_samples = int(max(1,len(value)*compression))
            heapqueue = []
            for i in range(len(nb_samples)):
                heapq.heappush(heapqueue,(backwards(value[i]),i))
            for i in range(nb_samples,len(value)):
                heapq.heappushpop(heapqueue,(backwards(value[i]),i))

            values, indices = [],[]
            for j in range(nb_samples,len(value)):
                v,ind = heapq.heappop(heapqueue)
                values.append(v)
                indices.append(ind)
            updates[var.op.name]=values
            indices_dict[var.op.name]=indices
        return pck.dumps((iteration,updates,indices_dict),protocol=-1)

    else :
        # Full matrix communications
        for var in sess.graph.get_collection_ref(collection):
            updates[var.op.name]= sess.run(var).flatten().tolist()
        return  pck.dumps([iteration,updates],protocol=-1)

def decode_variables(message):
    return pck.loads(message)
