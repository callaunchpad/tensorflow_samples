'''
Description: Trains neural network on mnist dataset to recognize handwritten digits

Details:
Layers = 3
Layer Sizes = [500, 500, 500]
Classes = 10 (digits 0-9)
Batch Size = 100
Activation Function = ReLU (Rectified Linear Unit)
Optimizer = AdamOptimizer (stochastic gradient descent)
Cost Function = softmax
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
# one_hot = True: 1 element in vector of classes (labels) is on, rest are off

0 = [1,0,0,0,0,0,0,0,0,0] # instead of 0 = 0
1 = [0,1,0,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
'''

# hyperparameters
# number of nodes per hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# number of classes
n_classes = 10
# number of data points fed into NN at a time
batch_size = 100

# Placeholder represents 28 x 28 pixel matrix as a flattened data point (height x width)
data = tf.placeholder('float', [None, 784]) # second parameter to catch malformed data errors
# Placeholder represents labels
label = tf.placeholder('float')

# data = raw_input_data
def neural_network_model(data):
    # weight vectors = num_inputs * num_nodes
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer =   {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                       'biases': tf.Variable(tf.random_normal([n_classes]))}

    # NN Model: (input_data * weight) + bias
    # Each layer runs activation function
    # ReLU speeds up learning (prevents vanishing gradient)
    l1 = tf.nn.relu(tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']))
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']))
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(data):
    prediction = neural_network_model(data)
    # cost function: calculates difference between prediction and known label (one_hot format)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, label))
    # Same as stochastic gradient descent and Ada grad
    # default learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + backpropagation
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # num of batches that encompass the num of total samples
        cycles = int(mnist.train.num_examples/batch_size)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(cycles):
                # moves through dataset in chunks
                epoch_data, epoch_label = mnist.train.next_batch(batch_size)
                # tf session modifies weights in layers
                _, c = sess.run([optimizer, cost], feed_dict = {data: epoch_data, label: epoch_label})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # check if the prediction was the same as the actual label
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({data: mnist.test.images, label: mnist.test.labels}))

train_neural_network(data)
