# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1, x2)

sess = tf.Session()
print sess.run(result)
sess.close()
