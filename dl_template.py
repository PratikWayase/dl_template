
# Import TensorFlow and Define Network Architecture

import tensorflow as tf
n_inputs = 28*28   images are 28x28 pixels
n_hidden1 = 300    # Neurons in the first hidden layer
n_hidden2 = 100    # Neurons in the second hidden layer
n_outputs = 10     # 10 classes (digits 0-9)

 =======================================

# placeholders for Inputs and Targets

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")  # Input data
y = tf.placeholder(tf.int64, shape=(None), name="y")              # Labels

# X is a 2D tensor where each row is an image 
# y is a 1D tensor containing the class labels 


 =======================================

#Define a function neuron_layer to create a layer of neurons:

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)  # Standard deviation for weight initialization
        W = tf.Variable(tf.truncated_normal((n_inputs, n_neurons), stddev=stddev), name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)  # ReLU activation
        else:
            return z

 #creates a layer with weights (W), biases (b), and an optional activation function 

 =======================================

# neuron_layer function to build the DNN:

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")  # No activation for logits

# note : logits is the output of the network before applying the softmax function.

 =======================================

# Define the Loss Function

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# sparse_softmax_cross_entropy_with_logits computes the cross-entropy directly from the logits.


 =======================================

# Define the Optimizer = Use Gradient Descent to minimize the loss:

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

 =======================================

# Evaluate the Model

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)  # Check if the top prediction is correct
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

 =======================================

#  save the model:

init = tf.global_variables_initializer()
saver = tf.train.Saver()

 =======================================

 ## Execution Phase ## 


data = input_data.read_data_sets("/tmp/data/")


 =======================================

# set the number of epochs and batch size:

n_epochs = 400
batch_size = 50


#train

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")



 =======================================

## usgin train mdoel

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = [...]  # New images (scaled from 0 to 1)
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)  # Predict the class with the highest logit