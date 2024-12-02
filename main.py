# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:50:36 2023

@author: 10947
"""

import warnings
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

warnings.filterwarnings("ignore")  

print('----start a lot of data sets---------')

file_name_ = ['coil100','Wafer']

for file_name in file_name_:

    print(file_name)
    data = get_data(file_name)
    k = 2048 #Generate a large probability list to ensure that the initial values and the starting probabilities are the same each time.
    
    
    rdx_name = 'save/' + file_name + '_' + 'rdx_' + str(k)
    rand_name = 'save/' + file_name + '_' + 'rand_' + str(k)
    
    if os.path.exists(rdx_name):
        print('exist')
        rdx = np.array([int(np.loadtxt(rdx_name))])
        rand_ = np.loadtxt(rand_name)
    else:
        print('no exist')
        n = data.shape[0]  
        rdx = np.random.choice(range(n), 1)   
        rand_ = list()   
        for i in range(k):
            rand_.append(random.random())
        np.savetxt(rdx_name,rdx)
        np.savetxt(rand_name,rand_)
    
    kk = [16]
    for i in range(len(kk)):
        k = kk[i]
        print(k)
        
        
        print('-----------CKM k-means++--------')
        s = time.time()
        centers,cal_dis_num = CKM(data,k,rdx,rand_)
        e = time.time()
        print(e-s)
        print(cal_dis_num)
        
        
        
        
        print('--------TI-------------')
        s = time.time()
        centers_inequality,cal_dis_num = TI(data,k,rdx,rand_)
        e = time.time()
        print(e-s)
        print(cal_dis_num)
        
        
        
        
        print('---------LBF_PAA-----------')
        block_size = 16
        s = time.time()
        centers_mean,cal_dis_num = LBF_PAA(data,k,block_size,rdx,rand_)
        e = time.time()
        print(e-s)
        print(cal_dis_num)
        
        
        
        print('---------TILB_PAA-----------')
        block_size = 16
        s = time.time()
        centers_mean_inequality,cal_dis_num = TILB_PAA(data,k,block_size,rdx,rand_)
        e = time.time()
        print(e-s)
        print(cal_dis_num)
        
        
        
        
        
        
        print('--------LBF_PPD------------')
        block_size = 16
        s = time.time()
        centers_partial,cal_dis_num = LBF_PPD(data,k,block_size,rdx,rand_)
        e = time.time()
        print(e-s)
        print(cal_dis_num)
        
        
        
        
        
        print('--------TILB_PPD------------')
        block_size = 16
        s = time.time()
        centers_partial_inequality,cal_dis_num = TILB_PPD(data,k,block_size,rdx,rand_)
        e = time.time()
        print(e-s)
        print(cal_dis_num)
        
        
        
        
        print('--------MCMC-----m=200-------')
        m = 200
        s = time.time()
        centers_MCMC = kmeans_MCMC(data,k,m,rdx)
        e = time.time()
        print(e-s)
        
        print('--------------------------------')
    
    
   


