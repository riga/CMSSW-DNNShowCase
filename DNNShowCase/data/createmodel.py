# -*- coding: utf-8 -*-

"""
Example of how a tfdeploy model is created from a tensorflow computation tree.
"""


import tfdeploy as td
import tensorflow as tf


# create the tensorflow tree
sess = tf.Session()
x = tf.placeholder("float", shape=[None, 3], name="input")
W = tf.Variable(tf.truncated_normal([3, 10], stddev=0.05))
b = tf.Variable(tf.zeros([10]))
y = tf.reduce_mean(tf.nn.relu(tf.matmul(x, W) + b), name="output")
sess.run(tf.initialize_all_variables())

# normally, this would be the right spot to create a cost function that uses labels
# and start the training but we skip this here for simplicity

# create the tfdeploy model
model = td.Model()
model.add(y, sess)
model.save("showcasemodel.pkl")
