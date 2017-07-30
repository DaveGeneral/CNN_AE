import tensorflow as tf
import argparse
from util import *
import numpy as np


class model():
    def __init__(self ,args):
        self.args = args
        self.inputs = tf.placeholder(dtype=tf.float32, [None, args.size, args.size, args.channel], 'Convolutional_Ae_inputs')
        
        enocoder_o = [self.inputs]
        for filter in args.filters:
            encoder_o.append(tf.layers.conv2d(encoder_o[-1], filter, [3,3], [2,2], padding='SAME', name='encoder_{}'.format(filter)))

        flatten_ = tf.contrib.layers.flatten(encoder_o)
        self.compression_feature = tf.layers.dense(flatten, args.compression_size, tf.nn.sigmoid, name='compression_dense')
        
        decoder_o = [self.compression_feature]
        for filter in args.filters[-2::-1]:
            decoder_o.append(tf.layers.conv2d_transpose(decoder_o[-1], filter, [3,3], [2,2], padding='SAME', name='decoder_{}'.format(filter)))
        decoder_o.append(tf.layers.conv2d_transpose(decoder_o[-1], 3, [3,3], [2,2], padding='SAME', name='decoder_{}'.format(3)))
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_o[-1], labels=self.inputs)

    def train(self):
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer.run()
            filewriter = tf.summary.FileWriter('logdir/', sess.graph)
            saver = tf.train.Saver(tf.global_variables())

            for itr in self.args.itrs:
                
                if itr % 100 == 0:
                    pass

                if itr % 1000 ==0:
                    saver.save(sess, 'saved')

if '__main__' == __name__:
    argparser = argparse.ArgumentParser()
    


