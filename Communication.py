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
        return not float.__le__(abs(self), abs(other))
    def __le__(self, other):
        return not float.__lt__(abs(self), abs(other))
    def __gt__(self, other):
        return not float.__ge__(abs(self), abs(other))
    def __ge__(self, other):
        return not float.__gt__(abs(self), abs(other))


def encode_variables(sess,collection,iteration,compression=1):
    updates = {}
    # Compression of variables
    if compression<1 :
        indices_dict = {}
        for var in sess.graph.get_collection_ref(collection):
            value = sess.run(var).flatten()
            nb_samples = int(max(1,len(value)*compression))
            heapqueue = []
            for i in range(len(value)):
                heapq.heappush(heapqueue,(backwards(value[i]),i))
            values, indices = [],[]
            for j in range(nb_samples):
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
