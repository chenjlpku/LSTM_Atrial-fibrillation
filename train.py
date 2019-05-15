# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:09:27 2019

@author: ChenJL
"""

from __future__ import print_function

from sklearn.preprocessing import normalize

import argparse
import time
import os
from six.moves import cPickle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#定义参数
# 数据参数
parser.add_argument('--data_dir',type=str,default='data/heartbeat',help='data input containing patients feature in time')

# 模型参数
parser.add_argument('--log_path',type=str,default='logs',help='directory to store tensorboard logs')
parser.add_argument('--save_path',type=str,default='save',help='directory to save checkpoint models')
parser.add_argument('--model',type=str,default='lstm',help='Name of the model')
parser.add_argument('--nums_layers',type=int,default=1,help='the layer of the Rnn model')
# 训练参数
parser.add_argument('--num_epoches',type=int,default=500, help = 'number of training epoch')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning_rate of training process')
parser.add_argument('--cell_size',type=int,default=256, help = 'size of rnn cell')
parser.add_argument('--num_class',type=int,default=4,help='number of classes')
parser.add_argument('--input_keep_prob',type=float,default=1,help='input keep prob')
parser.add_argument('--output_keep_prob',type=float,default=1,help='output keep prob')
parser.add_argument('--save_every',type=int,default=10,help='period of save a model')
parser.add_argument('--valid_step',type=int,default =10, help = 'period of validation')
# 优化参数
parser.add_argument('--batch_size',type=int,default=50,help='Minibach size')
parser.add_argument('--fea_length',type=int,default=150,help='number of features for each patient')


args = parser.parse_args()

import tensorflow as tf
from utils import DataLoader
from model import Model
import numpy as np
import time

def train(args):
    # 首先读入数据，其实读入的是一个data的方法集合
    data_loader = DataLoader(args.data_dir,args.batch_size,args.fea_length)
    # 从save_path中读取checkpoint
    if not os.path.exists(path=args.save_path):
        os.mkdir(path=args.save_path)
    
    train_writer = tf.summary.FileWriter(logdir=os.path.join(args.log_path,time.strftime("%Y-%m-%d-%H-%M-%S")+'-training'),graph=tf.get_default_graph())
    validation_writer = tf.summary.FileWriter(logdir=os.path.join(args.log_path,time.strftime("%Y-%m-%d-%H-%M-%S")+'-validation'),graph=tf.get_default_graph())
    with open(os.path.join(args.save_path,'config.pkl'),'wb') as f:
        cPickle.dump(args,f)
    
    model = Model(args)
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep = 5)
    
    with tf.Session() as sess:

        sess.run(init_op)
        # 从checkpoint中载入模型
        checkpoint = tf.train.get_checkpoint_state(args.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path+'.meta')
            saver.restore(sess,tf.train.latest_checkpoint(args.save_path))
        # 模型载入完毕
        # 开始按照epoch进行训练
        for e in range(args.num_epoches):
            data_loader.reset_batch_point() # 每一轮开始都要重新设置batch的起点
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x,y = data_loader.next_batch
                x[np.isnan(x)] = 0
                feed  = {model.input_data:x, model.target:y}
                summ, train_accuracy,state,_ = sess.run([model.merge,model.accuracy,model.final_state,model.train_op],feed_dict = feed)
                train_writer.add_summary(summ,e*data_loader.num_batches+b)
                end = time.time()
                
                print("{}/{} (epoch {}), accuracy={:.3f},time/batch = {:.3f}".format(e*data_loader.num_batches+b,args.num_epoches*data_loader.num_batches,e,train_accuracy,end-start))
                # 保存模型
                if (e*data_loader.num_batches+b)%args.save_every == 0 or (e == args.num_epoches-1  and b == data_loader.num_batches-1):
                    checkpoint_path = os.path.join(args.save_path,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step = e*data_loader.num_batches+b)
                    print("model saved to {}".format(checkpoint_path))
                
            if e % args.valid_step == 0 and e>0:
                xv,yv = data_loader.validation_data()
                state_v = np.zeros([args.batch_size,args.cell_size])
                feed_v = {model.input:xv,model.targets:yv,model.initial_state:state_v}
                summ_v,valid_loss,_ = sess.run([model.merge,model.loss,model.train_op],feed_v)
                validation_writer.add_summary(summ_v,e)
                print('epoch {}, valid_loss={:.3f}'.format(e,valid_loss))
                
            
if __name__ == '__main__':
    train(args)
    # trian.py 的设计分成几步：1、定义参数，2、载入数据，3、尝试载入模型，4、分epoch和batch进行训练，并将结果写入log