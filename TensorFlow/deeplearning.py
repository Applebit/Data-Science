#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 10:14:15 2016

@author: root-007
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    input > weight > hidden layer 1(activation function) > weight ....so on till output layer
    
    compare output to intended output >cost function
    optmization function > minimize cost > minimize cost
    
    backpropagation
    
    feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
# 10 class 0 to 9
'''
 One hot means applicable element vector is on all false
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
                      }
    hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
                      }
    hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
                      }
                      
    output_layer =   {'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))
                      }

                      
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weight']) + output_layer['biases']

    return output
    
def train_neural_network(x):
    global y
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print 'Epoch', epoch, ' completed out of ', hm_epochs, 'loss:', epoch_loss
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
        accurecy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy', accurecy.eval({x:mnist.test.images, y:mnist.test.labels})
        
train_neural_network(x)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    