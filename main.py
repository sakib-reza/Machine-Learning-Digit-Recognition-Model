# importing data set
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# importing image manipulation libraries
import numpy as np
from PIL import Image

mnist = input_data.read_data_sets("MNIST_dataset/", one_hot=True)
# seperating training/validation/test images
n_training = mnist.train.num_examples  # 55,000
n_validating = mnist.validation.num_examples  # 5000
n_testing = mnist.test.num_examples  # 10,000

# defining neural network
input_layer = 784  # 28x28 pixels
hidden_layer1 = 512  
hidden_layer2 = 256  
hidden_layer3 = 128  
output = 10  # 0-9 digits
learning_rate = 1e-4
iteration_count = 1000
group_size = 256
dropout = 0.5

# building tensorflow graph
Xaxis = tf.placeholder("float", [None, input_layer])
Yaxis = tf.placeholder("float", [None, output])
retain_probability = tf.placeholder(tf.float32)
weights = {
    'w1': tf.Variable(tf.truncated_normal([input_layer, hidden_layer1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([hidden_layer3, output], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[hidden_layer1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[hidden_layer2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[hidden_layer3])),
    'out': tf.Variable(tf.constant(0.1, shape=[output]))
}
layer1 = tf.add(tf.matmul(Xaxis, weights['w1']), biases['b1'])
layer2 = tf.add(tf.matmul(layer1, weights['w2']), biases['b2'])
layer3 = tf.add(tf.matmul(layer2, weights['w3']), biases['b3'])
drop_layer = tf.nn.dropout(layer3, retain_probability)
output_layer = tf.matmul(layer3, weights['out']) + biases['out']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Yaxis, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# training the model
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Yaxis, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# train in groups
for i in range(iteration_count):
    batch_x, batch_y = mnist.train.next_batch(group_size)
    sess.run(train_step, feed_dict={
        Xaxis: batch_x, Yaxis: batch_y, retain_probability: dropout
        })
    # printing accuracy and loss per group
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={Xaxis: batch_x, Yaxis: batch_y, retain_probability: 1.0}
            )
        print("Run Count:", str(i), " | Loss =", str(minibatch_loss), " | Accuracy =", str(minibatch_accuracy))

#overall test accuracy
test_accuracy = sess.run(accuracy, feed_dict={Xaxis: mnist.test.images, Yaxis: mnist.test.labels, retain_probability: 1.0})
print("\nAccuracy of the model is:", test_accuracy)

# for testing images not in the data set
"""
img = np.invert(Image.open("test_img.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))
"""