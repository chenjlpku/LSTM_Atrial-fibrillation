# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:09:47 2019

@author: ChenJL
"""

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

class Model():
    def __init__(self,args,training = True):
        self.args = args
        
        if not training:
            args.batch_size = 1
        
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception('model type not supported :{}'.format(args.model))
        
        cells = []
        for _ in range(args.nums_layers):
            cell = cell_fn(args.cell_size)
            if training and (args.output_keep_prob<1.0 or args.input_keep_prob < 1.0):
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob = args.input_keep_prob,output_keep_prob = args.output_keep_prob)
            cells.append(cell)
        
        self.cell = cell = rnn.MultiRNNCell(cells,state_is_tuple = True)

        self.input_data = tf.placeholder(tf.float32,[args.batch_size,args.fea_length]) # 这里，args.seq_length在场景中代表着样本的特征点数
        self.target = tf.placeholder(tf.float32,[args.batch_size,args.num_class])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        
        input_data_expand = tf.expand_dims(self.input_data,-1)
        self.outputs, states = tf.nn.dynamic_rnn(self.cell,input_data_expand,initial_state = self.initial_state,dtype = tf.float32)
        # 此时的outputs的纬度是（batch_size,seq_len,cell_size）,states的维度是(batch_size,cell_size)
        
        
        states = tf.identity(states,name = 'H')
        # 计算softmax层的输入和输出
        with tf.name_scope('final_training_ops'):
            with tf.name_scope('weight'):
                weight = tf.get_variable('weights',initializer=tf.random_normal([args.cell_size,args.num_class],stddev = 0.01))
            with tf.name_scope('biases'):
                bias = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[args.num_class]))           
            hf = tf.transpose(self.outputs,[1,0,2])
            self.last = tf.gather(hf,int(hf.get_shape()[0])-1)
            with tf.name_scope('logits'):
                self.logits = tf.matmul(self.last,weight)+bias
            with tf.name_scope('Prop'):
                self.prop = tf.nn.softmax(self.logits)
        # 计算损失，accuracy, 和优化器
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.target))
            self.loss = loss
            tf.summary.scalar('loss',loss)
        
        self.final_state = states
        
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.arg_max(self.prop,1),tf.argmax(self.target,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            self.accuracy = accuracy
            tf.summary.scalar('accuracy',accuracy)
            
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
            self.train_op = optimizer
        
        self.merge = tf.summary.merge_all()