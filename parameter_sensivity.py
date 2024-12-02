# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:03:14 2023

@author: 10947
"""

from tool import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from kmeans_plus_plus import *
from data import *
import time
import os
from preprocessing import *




file_name = 'mixed_sinx'
print(file_name)
data = get_data(file_name)
k = 2048 


rdx_name = 'save/' + file_name + '_' + 'rdx_' + str(k)
rand_name = 'save/' + file_name + '_' + 'rand_' + str(k)

if os.path.exists(rdx_name):
    print('exist')
    rdx = np.array([int(np.loadtxt(rdx_name))])
    rand_ = np.loadtxt(rand_name)
else:
    print('no exist')
    n = data.shape[0]  
    rdx = np.random.choice(range(n), 1)   #每一个的初始化都一样
    rand_ = list()   #保存每次轮盘法的旋转，，使得运行结果一致
    for i in range(k):
        rand_.append(random.random())
    np.savetxt(rdx_name,rdx)
    np.savetxt(rand_name,rand_)

block_vector_all = [2,4,8,16,32,64,128]
k = 32
time_save_mean = list()
time_save_partial = list()
time_save_mean_inequality = list()
time_save_partial_inequality = list()
dis_save_mean = list()
dis_save_mean_inequality = list()
for block_vector_ in block_vector_all:
    
    print(block_vector_)
    
    
    
    
    print('---------LBF_PAA-----------')
    block_size = block_vector_
    s = time.time()
    centers_mean,cal_dis_num = LBF_PAA(data,k,block_size,rdx,rand_)
    e = time.time()
    print(e-s)
    print(cal_dis_num)
    time_save_mean.append(e-s)
    dis_save_mean.append(cal_dis_num)
    
    
    
    print('---------TILB_PAA-----------')
    block_size = block_vector_
    s = time.time()
    centers_mean_inequality,cal_dis_num = TILB_PAA(data,k,block_size,rdx,rand_)
    e = time.time()
    print(e-s)
    print(cal_dis_num)
    time_save_mean_inequality.append(e-s)
    dis_save_mean_inequality.append(cal_dis_num)
   
    
    
    
    
    
    print('--------LBF_PPD------------')
    block_size = block_vector_
    s = time.time()
    centers_partial,cal_dis_num = LBF_PPD(data,k,block_size,rdx,rand_)
    e = time.time()
    print(e-s)
    print(cal_dis_num)
    
    time_save_partial.append(e-s)
    
    
    
    print('--------TILB_PPD------------')
    block_size = block_vector_
    s = time.time()
    centers_partial_inequality,cal_dis_num = TILB_PPD(data,k,block_size,rdx,rand_)
    e = time.time()
    print(e-s)
    print(cal_dis_num)
   
    time_save_partial_inequality.append(e-s)
    
       
    print('--------------------------------')







