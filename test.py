# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:10:03 2019

@author: ChenJL
"""

import argparse
import os
import cPickle
import numpy as np
import pickle

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--save_path',type = str, default = 'save',help='directory to save checkpoint models')
parser.add_argument('--cell_size',type = int, default=256,help='size of rnn cell')
parser.add_argument('--data_dir',type=str,default='data/heartbeat',help='data input containing patients feature in time')

args = parser.parse_args()

import tensorflow as tf
from model import Model

def test(args,test_data):
    with open(os.path.join(args.save_path,'config.pkl'),'rb') as f:
        save_args = cPickle.load(f)
    
    model = Model(save_args,training = False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(args.save_path)
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path+'.meta')
        saver.restore(sess,tf.train.latest_checkpoint(args.save_path))
        
        state_t = np.zeros([1,args.cell_size])
        feed = {model.input_data:test_data,model.initial_state:state_t}
        test_prop = sess.run(model.prob,feed)
        
        print("The test result is {}".format(test_prop))

if __name__ == '__main__':
    test_data_add = os.path.join(args.data_dir,'TestData.p')
    with open(test_data_add,'rb') as f:
        test_data = pickle.dump(f)
    test(args,test_data)