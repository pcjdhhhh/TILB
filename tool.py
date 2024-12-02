import numpy as np
import math
import matplotlib.pyplot as plt
from preprocessing import *

   
def calEuclidean(a,b):
    
    length = len(a)
    sum_ = 0
    for i in range(length):
        sum_ = sum_ + (a[i]-b[i])**2
    return sum_


def get_nearest_dist_using_pair_small_lower_bound_block(point,new_center,new_index,current_dis,current_index,block_size,point_norm,point_block,center_norm,center_block):
    
    cal = 0
    #lower_bound = calEuclidean(mean_info,center_info)*block_size
    lower_bound = 0
    length = len(center_block)
    for i in range(length):
        #print(i)
        lower_bound = lower_bound + point_block[i]*center_block[i]
    lower_bound = point_norm+center_norm - 2*lower_bound
    if lower_bound<current_dis:
        cal = 1
        temp_dis = calEuclidean(point,new_center)
        if temp_dis<current_dis:
            
            current_dis = temp_dis
            current_index = new_index
    
    return current_dis,current_index,cal


def get_nearest_dist_using_pair_small_lower_bound(point,new_center,new_index,current_dis,current_index,block_size,mean_info,center_info):
    
    cal = 0
    lower_bound = calEuclidean(mean_info,center_info)*block_size
    if lower_bound<current_dis:
        cal = 1
        temp_dis = calEuclidean(point,new_center)
        if temp_dis<current_dis:
            
            current_dis = temp_dis
            current_index = new_index
    
    return current_dis,current_index,cal


def get_nearest_dist_using_pair_small(point,new_center,new_index,current_dis,current_index):
    
    temp_dis = calEuclidean(point,new_center)
    if temp_dis<current_dis:
       
        current_dis = temp_dis
        current_index = new_index
    return current_dis,current_index

def generate_parameters(dim):
    
    max_ = math.floor(dim/2)
    res = list()
    first = 2
    while True:
        if first<=max_:
            res.append(first)
            first = first * 2
        else:
            break
    return res

def get_nearest_dist_brute_force(point,centers):
    
    cal_dis_num = 0
    n_center = centers.shape[0]
    min_dis = math.inf
    for i in range(n_center):
        temp_center = centers[i,:]
        cal_dis_num = cal_dis_num + 1
        temp_dist = calEuclidean(point,temp_center)
        if temp_dist<min_dis:
            min_dis = temp_dist
    return min_dis,cal_dis_num
    
def get_nearest_dist_inequality(point,centers):
    
    cal_dis_num = 0
    index = -1
    n_center = centers.shape[0]
    min_dis = math.inf
    for i in range(n_center):
        temp_center = centers[i,:]
        cal_dis_num = cal_dis_num + 1
        temp_dist = calEuclidean(point,temp_center)
        if temp_dist<min_dis:
            index = i
            min_dis = temp_dist
    return min_dis,cal_dis_num,index