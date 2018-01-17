# AdaComp

AdaComp is a method to distribute stochastic gradient descent (SGD) onto a large number of workers in the parameter server model (PS). The AdaComp method is described in paper:

"Distributed deep learning on edge-devices: feasibility via adaptive compression". Corentin Hardy, Erwan Le Merrer and Bruno Sericola. In IEEE NCA 2017.

## List of files

 * PS.py
 * Worker.py
 * Supervisor.py
 * Tf_op.py
 * Communications.py
 * main.py
 * mnist.py
 * data_Mnist.tar.gz

## Getting Started

This code has to be placed on different interconnected computational nodes:
- One of them has the role of the PS. It communicates with every other nodes to maintain a central model.
- By default, other nodes are workers. They compute updates for the the central model using their local data. Updates are compressed before sending. 
- Optionally, an other node could have the role of Supervisor. It contains a test dataset with which it computes the accuracy of the central model.


### Prerequisites

AdaComp requires :

* Python 2.7
* Tensorflow (>=1.0)
* Numpy 1.13

### Running Deep MNIST example

The example framework trains a 4-layers CNN with the same architecture than [mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) in Keras examples.


	python main.py 	[ --iter_max argument ] [ --type_node argument ] [ --id_worker argument ]
			[ --nb_workers argument ][ --ip_PS argument ][ --port argument ]
			[ --image_size argument ][ --image_depth argument ][ --nb_classes argument ]
			[ --batch_size argument ][ --compression_rate argument ][ --data_dir argument ]
			[ --learning_rate ]


Description of possibles flags :

	--iter_max (default: 100000)   		Number of iterations to run on the PS.
	--type_node (default: 'Worker')   	Worker|PS|Supervisor : define the role of the node executing the code.
	--nb_workers (default: 100)   		Number of workers.
	--id_worker (default: 1)   		ID of the local worker (should be included between 1 and nb_workers).
	--ip_PS (default: '0.0.0.1')   		The ip address of PS.
	--port (default: 2223)   		The port used in PS.
	--image_size (default: 28)   		The size of an image.
	--image_depth (default: 1)   		The depth of an image.
	--nb_classes (default: 10)   		Number of classes.
	--batch_size (default: 10)		Number of images to process in a batch.
	--compression_rate (default: 0.01)	Compression rate of worker updates.
	--data_dir (default: './')   		Path to the MNIST data directory.
	--learning_rate (default: 0.1)   	The learning rate of the Gradient Optimizer


The MNIST training dataset contained in data_Mnist.tar.gz was split in 1200 files of 50 images each called "train-images-xxxxx.idx3-ubyte" for images and "train-labels-xxxxx.idx1-ubyte" for labels. This training files have to be distributed on workers; e.g, worker with ID `n` has files `(n-1)x[1200/nb_workers]` to `n x [1200/nb_workers]`. The MNIST test dataset has to be placed on Supervisor node.

To run the script with default hyper-parameters and 10 workers :

* On PS node :

		python main.py --type_node PS --nb_workers 10

* On Worker node `n` :

		python main.py --nb_workers 10 --id_worker n --ip_PS xxxxxxx --data_dir path/to/trainning/dataset/directory

* On Supervisor node :

		python main.py --type_node Supervisor --nb_workers 10 --batch_size 100000 --data_dir path/to/test/dataset/directory


### Add your own model

Your own model should be specified in file PS.py, Worker.py and Supervisor.py using a Tensorflow graph. Moreover you have to manage your dataset inputs and labels. Trainable variables shared by all workers via the PS have to be put in a Tensorflow collection called "W_global".

The main functions are :

- `Worker(D,graph=None)` in Worker.py
	* Args :
		- D : Training dataset available locally
		- graph (optional) : current graph used for data preprocessing.
	* Perform worker iterations until the number of iteration is greater than iter_max.
- `PS()` in PS.py
	* Update the model using worker gradients until the number of iteration is greater than iter_max + nb_workers-1.

* Other files :
  - Supervisor.py contains a function to compute accuracy and loss of the model on a test dataset.
  - Tf_op.py contains functions to build the tensorflow operation used to update the model.
  - Communication.py contains functions to compress, send and receive parameters of the model.

## Optimization

For clarity reasons, the code of the PS has been simplified to use only one thread. For better performance, multithreading must be used in the PS to process multiple worker requests in parallel.

## Author

* Corentin Hardy
