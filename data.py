# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:12:18 2022

@author: 10947
"""
import pandas
import random
from scipy import stats
from scipy.io import loadmat
from PIL import Image
import math
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
import os
from tool import *


def get_mixed_sinx(n=100000,length=256):
    '''
    data = np.zeros((n,length))
    t = np.linspace(0,1,length)
    for i in range(n):
        res = np.zeros(length)
        am = random.uniform(1, 10)
        f = random.uniform(1,10)
        res = am*np.sin(2*math.pi*f*t)
        data[i,:] = res
    return data
    '''
    path = 'datasets/sinx/sinx'
    return np.loadtxt(path)





def get_mfeat():
    path = 'datasets/mfeat' + '/' + 'mfeat.mat'
    data = loadmat(path)
    res = data['mfeat']
    return res

def get_time_series(file_name):
    
    #file_name = 'Car'
    train_file_path = 'datasets/' + 'time_series/' + file_name + '/' + 'train.mat'
    test_file_path = 'datasets/' + 'time_series/' + file_name + '/' + 'test.mat'
    train = loadmat(train_file_path)
    test = loadmat(test_file_path)
    train = train['train']
    test = test['test']
    train_label = train[:,0]   
    test_label = test[:,0]     
    train_data = train[:,1:]   
    test_data = test[:,1:]     
   
    [num,dim] = train_data.shape
    for i in range(num):
        temp = stats.zscore(train_data[i],ddof=1)
        train_data[i] = temp
    
    [num,dim] = test_data.shape
    for i in range(num):
        temp = stats.zscore(test_data[i],ddof=1)
        test_data[i] = temp
    res = np.vstack((train_data,test_data))
    return res



def get_usps():
    path = 'datasets/usps' + '/' + 'usps.mat'
    data = loadmat(path)
    return data['fea'].astype(float)

def get_coil100():
    path = 'datasets/coil100/COIL100.mat'
    data = loadmat(path)
    return data['fea'].astype(float)





def get_gaussin(n=100000,length=128):
    '''
    data = np.zeros((n,length))
    for i in range(n):
        mu = random.randint(-10,10)
        std = random.randint(1,5)
        res = np.zeros(length)
        for j in range(length):
            res[j] = random.gauss(mu,std)
        data[i,:] = res
    return data
    '''
    path = 'datasets/gaussin/gaussin'
    return np.loadtxt(path)
    




def get_data(file_name):
    if file_name=='usps':
        return get_usps()
    elif file_name=='gaussin':
        return get_gaussin()
    elif file_name=='coil100':
        return get_coil100()
    elif file_name=='mfeat':
        return get_mfeat()
    elif file_name=='mixed_sinx':
        return get_mixed_sinx()
    else:
        return get_time_series(file_name)
    return 0



