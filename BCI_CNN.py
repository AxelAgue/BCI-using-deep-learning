import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split


f1 = h5py.File('/home/oem/data/EEG_ParticipantA_28_09_2013.mat','r')
data1 = f1.get("data_epochs_A")
label1 = f1.get("data_key_A")
data1 = np.array(data1)
label1 = np.array(label1)

f2 = h5py.File('/home/oem/data/EEG_ParticipantB_30_09_2013.mat','r')
data2 = f2.get("data_epochs_B")
label2 = f2.get("data_key_B")
data2 = np.array(data2)
label2 = np.array(label2)


"""
dataa = []

for i in range(len(data2)):
    new_data2 = data2[i]
    dataaa = []
    for j in range(len(new_data2)):
        new_data2[j] = new_data2[j] + 2006.3681663
        dataaa.append(new_data2)
    dataa.append(dataaa)
"""
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]




from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
new_data = mms.fit_transform(data2)

#print(len(data2[0]))
data2_crop = crop_center(new_data, 784, 1377)
#print(len(data2_crop[0]))
#exit()



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
new_label = ohe.fit_transform(label2)


X_train, X_test, y_train, y_test = train_test_split(data2_crop, new_label, test_size=0.20, random_state=0)

train_data = X_train
train_data_labels = y_train


test_data = X_test
test_data_labels = y_test

x = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")
y_ = tf.placeholder(tf.float32, shape=[None, 3], name="outputs")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5, name="Weights")
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name="bias")
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
with tf.name_scope("inputs"):
    tf.summary.image("inputs", x_image, 3)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
with tf.name_scope("Parameters"):
    tf.summary.histogram("Weights_1", W_conv1)
    tf.summary.histogram("Bias_1", b_conv1)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
with tf.name_scope("Parameters"):
    tf.summary.histogram("Weights_2", W_conv2)
    tf.summary.histogram("Bias_2", b_conv2)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = weight_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope("corss_entropy"):
    tf.summary.scalar("corss_entropy", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

merged_summary = tf.summary.merge_all()
logs_path = "/home/oem/PycharmProjects/CNN_pruebas/Modelo1"

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logs_path)
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(10000):

        #mnist.train.images = tf.image.random_contrast(mnist.train.images, lower=0.2, upper=1.8)
        batch = train_data
        batch_labels = train_data_labels
        train_step.run(feed_dict={x: batch, y_: batch_labels, keep_prob: 0.5})

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: train_data, y_: train_data_labels, keep_prob: 1.0})
            print('step %d, train error %g' % (i, 1-train_accuracy))

            test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_data_labels, keep_prob: 1.0})
            print('step %d, test error %g' % (i, 1 - test_accuracy))


    #print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}))
