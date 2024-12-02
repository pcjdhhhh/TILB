# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:05:51 2022

@author: 10947
"""
#用于各种下界的预处理
#from search_function import *
import numpy as np
from tool import *
#from get_data import *
from sklearn.decomposition import PCA
import math
from collections import defaultdict

def preprocessing_with_PPD(train):
    pca = PCA()
    X_r = pca.fit(train).transform(train)
    e = pca.components_
    ref = e * (-1)
    return X_r,ref,pca
    






    

def preprocessing_with_mean(train,block_size):
    [n_train,dim] = train.shape
    if dim%block_size == 0:
        #可以整除,则
        block_num = dim // block_size  #块数
    else:
        #补零使其能够整除
        block_num = math.floor(dim / block_size) + 1
        should_add = block_size - (dim % block_size)
        #new_dim = dim + should_add
        add_train = np.zeros((n_train,should_add))
        train = np.hstack((train,add_train))  #矩阵末尾补零
    [n_train,dim] = train.shape
    mean_info = np.zeros((n_train,block_num))
    for i in range(n_train):
        current = train[i,:]
        #norm_info[i] = np.linalg.norm(current)**2
        for j in range(block_num):
            s = j*block_size
            e = (j+1)*block_size
            mean_info[i,j] = np.mean(current[s:e])
    return mean_info






            
        
        
        
    