# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:10:03 2019

@author: ChenJL
"""

import os
import pickle
import numpy as np

class DataLoader():
    def __init__(self,data_dir,batch_size,fea_length,encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fea_length = fea_length
        self.encoding = encoding
    
        self.train_data_add = os.path.join(data_dir,'TrainData.p')
        self.train_label_add = os.path.join(data_dir,'TrainLabel.p')
        
        self.valid_data_add = os.path.join(data_dir,'ValidData.p')
        self.valid_label_add = os.path.join(data_dir,'ValidLabel.p')
        
        self.load_data()
        
        self.create_batch()
        self.reset_batch_point()

    
    def load_data(self):
        with open(self.train_data_add,'rb') as ft:
            self.train_data = np.array(pickle.load(ft))
        with open(self.train_label_add,'rb') as ftl:
            self.train_label = np.array(pickle.load(ftl))
        with open(self.valid_data_add,'rb') as vt:
            self.valid_data = np.array(pickle.load(vt))
        with open(self.valid_label_add,'rb') as vtl:
            self.valid_label = np.array(pickle.load(vtl))
    
    def create_batch(self):
        self.num_batches = int(len(self.train_data)/self.batch_size)
        self.train_data = self.train_data[:self.num_batches*self.batch_size]
        self.train_label = self.train_label[:self.num_batches*self.batch_size]
        self.td_batch = np.split(self.train_data,self.num_batches,0)
        self.tl_batch = np.split(self.train_label,self.num_batches,0)
    
    def next_batch(self):
        x,y = self.td_batch[self.pointer],self.tl_batch[self.pointer]
        self.pointerdata  += 1
        return x,y
    def reset_batch_point(self):
        self.pointer = 0
    def validation_data(self):
        x,y = self.valid_data[:self.batch_size],self.valid_label[:self.batch_size]
        return x,y