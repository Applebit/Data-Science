#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:01:46 2016

@author: root-007
"""

import tensorflow as tf
from owndata_deeplearning import create_feature_sets_and_labels
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('positive.txt','negative.txt')
import numpy as np

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2

batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
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
#            for _ in range(int(mnist.train.num_examples/batch_size)):
#                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                epoch_loss += c
                i += batch_size
            print 'Epoch', epoch, ' completed out of ', hm_epochs, 'loss:', epoch_loss
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
        accurecy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy', accurecy.eval({x:test_x, y:test_y})
        
train_neural_network(x)