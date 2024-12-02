# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 10:39:47 2022

@author: 10947
"""
from tool import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from preprocessing import *


def kmeans_MCMC(dataset,k,m,rdx):
    #利用MCMC采样得到最终的聚类中心
    #论文：Approximate K-Means++ in Sublinear Time
    #参数m为MCMC采样中
    n = dataset.shape[0]  
    dim = dataset.shape[1]
    new_add_index = rdx[0]
    centers = dataset[rdx]  #初始的
    for temp_k in range(1,k):
        #从数据集dataset中随机采样一个x
        index_x = np.random.choice(range(n), 1)[0]
        x = dataset[index_x,:]
        dx,cal = get_nearest_dist_brute_force(x,centers)
        for j in range(1,m):
            index_y = np.random.choice(range(n), 1)[0]
            y = dataset[index_y,:]
            dy,cal = get_nearest_dist_brute_force(y,centers)
            if dy/dx > np.random.uniform(0,1):
                x = y
                dx = dy
        
        centers=np.vstack((centers,x))
    return centers
        


def LBF_PPD(dataset,k,block_size,rdx,rand_):
    #dataset:数据，n*dim的array
    #k：聚类的个数
    #利用partial_lower_bound加速
    #mean_info = preprocessing_with_mean(dataset,block_size)
    n = dataset.shape[0]  
    dim = dataset.shape[1]
    num_of_block = math.ceil(dim/block_size)
    #rdx = np.random.choice(range(n), 1)
    #rdx = np.array([save_index[0]])  #为了保证两次运行kmeans++结果一致
    new_add_index = rdx[0]
    centers = dataset[rdx]  #初始的
    #centers = np.vstack((centers,dataset[1,:]))
    d = np.array(np.ones(n)*np.inf)  #用于使用轮盘法计算概率，d[i]为第i个点与其最近中心的距离
    index = np.array(np.ones(n)*0)
    cal_dis_num = 0  #用于保存计算欧氏距离的次数
    #cal_dis_num_no_lower_bound = 0#用于保存不用lower——bound需要的计算距离次数
    for temp_k in range(0,k-1):
        tot = 0 #用于轮盘法求概率
        
        for i,point in enumerate(dataset):
            #判断point_i与新加入的聚类中心的lower_bound，
            #只有当lower_bound<d[i]的时候需要计算真实欧氏距离
            #lower_bound = calEuclidean(mean_info[i,:],mean_info[new_add_index,:]) * block_size
            lower_bound = 0
            flag = 0
            for j in range(num_of_block):
                start_index = j*block_size
                end_index = min(dim,(j+1)*block_size)
                lower_bound += calEuclidean(point[start_index:end_index],centers[len(centers)-1,start_index:end_index])
                if lower_bound>d[i]:
                    flag = 1
                    break
            
            if flag==0:
                d[i],index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,d[i],index[i])
                cal_dis_num = cal_dis_num + 1
                tot = tot + d[i]
            else:
                tot = tot + d[i]
        tot = tot * rand_[temp_k]   #转动轮盘
        for i, di in enumerate(d):
            tot -= di
            if tot > 0:
                continue
            centers=np.vstack((centers,dataset[i,:]))
            new_add_index = i  #新加入的center的下标
            break
        #centers=np.vstack((centers,dataset[save_index[temp_k+1],:]))
    return centers,cal_dis_num

def TILB_PPD(dataset,k,block_size,rdx,rand_):
    n = dataset.shape[0]  
    dim = dataset.shape[1]
    num_of_block = math.ceil(dim/block_size)
    #rdx = np.random.choice(range(n), 1)
    #rdx = np.array([save_index[0]])  #为了保证两次运行kmeans++结果一致
    new_add_index = rdx[0]
    centers = dataset[rdx]  #初始的
    #centers = np.vstack((centers,dataset[1,:]))
    d = np.array(np.ones(n)*np.inf)  #用于使用轮盘法计算概率，d[i]为第i个点与其最近中心的距离
    index = np.array(np.ones(n)*0)
    cal_dis_num = 0  #用于保存计算欧氏距离的次数
    #cal_dis_num_no_lower_bound = 0#用于保存不用lower——bound需要的计算距离次数
    for temp_k in range(0,k-1):
        #执行k-1次聚类
        center_to_center = np.zeros(temp_k+1)  #用于保存
        for i in range(temp_k+1):
            #center_to_center[i]为新加入的聚类中心与之前已经加入的聚类中心之间的距离
            #centers[temp_k,:]为新加入的聚类中心
            center_to_center[i] = calEuclidean(centers[temp_k,:],centers[i,:])
        tot = 0 #用于轮盘法求概率
        for i,point in enumerate(dataset):
            #首先使用三角不等式判断是否需要计算距离
            #index[i]为第i个点对应的聚类中心的下标
            #center_to_center[int(index[i])]为新加入的聚类中心与第i个点对应的聚类中心
            #之间的距离
            if np.sqrt(center_to_center[int(index[i])])/2.0 > np.sqrt(d[i]):   #注意：这里要有根号才满足
                #第i个点对应的中心仍然为index[i]
                tot = tot + d[i]
                #continue
            else:
                lower_bound = 0
                flag = 0
                for j in range(num_of_block):
                    start_index = j*block_size
                    end_index = min(dim,(j+1)*block_size)
                    lower_bound += calEuclidean(point[start_index:end_index],centers[len(centers)-1,start_index:end_index])
                    if lower_bound>d[i]:
                        flag = 1
                        break
                
                if flag==0:
                    d[i],index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,d[i],index[i])
                    cal_dis_num = cal_dis_num + 1
                    tot = tot + d[i]
                else:
                    tot = tot + d[i]
        tot = tot * rand_[temp_k]   #转动轮盘
        for i, di in enumerate(d):
            tot -= di
            if tot > 0:
                continue
            centers=np.vstack((centers,dataset[i,:]))
            #save_index.append(i)
            break
    return centers,cal_dis_num


def TI(dataset,k,rdx,rand_):
    #dataset:数据，n*dim的array
    #k：聚类的个数
    #save_index = list()#用于保存每次加进去的聚类中心的下标
    n = dataset.shape[0]  
    #rdx = np.random.choice(range(n), 1)
    #add_index = rdx[0]
    #save_index.append(add_index)
    centers = dataset[rdx]  #初始的
    #centers = np.vstack((centers,dataset[1,:]))
    d = np.array(np.ones(n)*np.inf)  #用于使用轮盘法计算概率，d[i]为第i个点与其最近中心的距离
    index = np.array(np.ones(n)*0)
    cal_dis_num = 0  #用于保存计算欧氏距离的次数
    #print(centers.shape)
    for temp_k in range(0,k-1):
        #执行k-1次聚类
        center_to_center = np.zeros(temp_k+1)  #用于保存
        for i in range(temp_k+1):
            #center_to_center[i]为新加入的聚类中心与之前已经加入的聚类中心之间的距离
            #centers[temp_k,:]为新加入的聚类中心
            center_to_center[i] = calEuclidean(centers[temp_k,:],centers[i,:])
        tot = 0 #用于轮盘法求概率
        for i,point in enumerate(dataset):
            #首先使用三角不等式判断是否需要计算距离
            #index[i]为第i个点对应的聚类中心的下标
            #center_to_center[int(index[i])]为新加入的聚类中心与第i个点对应的聚类中心
            #之间的距离
            if np.sqrt(center_to_center[int(index[i])])/2.0 > np.sqrt(d[i]):   #注意：这里要有根号才满足
                #第i个点对应的中心仍然为index[i]
                tot = tot + d[i]
                #continue
            else:
                #d[i],dis_num,index[i] = get_nearest_dist_inequality(point,centers)
                d[i],index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,d[i],index[i])
                cal_dis_num = cal_dis_num + 1
                tot = tot + d[i]
        
        tot = tot * rand_[temp_k]   #转动轮盘
        for i, di in enumerate(d):
            tot -= di
            if tot > 0:
                continue
            centers=np.vstack((centers,dataset[i,:]))
            #save_index.append(i)
            break
    return centers,cal_dis_num

def TILB_PAA(dataset,k,block_size,rdx,rand_):
    #结合三角不等式与距离下界的算法
    mean_info = preprocessing_with_mean(dataset,block_size)
    n = dataset.shape[0]
    new_add_index = rdx[0]
    centers = dataset[rdx]  #初始的
    #centers = np.vstack((centers,dataset[1,:]))
    d = np.array(np.ones(n)*np.inf)  #用于使用轮盘法计算概率，d[i]为第i个点与其最近中心的距离
    index = np.array(np.ones(n)*0)
    cal_dis_num = 0  #用于保存计算欧氏距离的次数
    for temp_k in range(0,k-1):
        #执行k-1次聚类
        center_to_center = np.zeros(temp_k+1)  #用于保存
        for i in range(temp_k+1):
            #center_to_center[i]为新加入的聚类中心与之前已经加入的聚类中心之间的距离
            #centers[temp_k,:]为新加入的聚类中心
            center_to_center[i] = calEuclidean(centers[temp_k,:],centers[i,:])
        tot = 0 #用于轮盘法求概率
        for i,point in enumerate(dataset):
            #首先使用三角不等式判断是否需要计算距离
            #index[i]为第i个点对应的聚类中心的下标
            #center_to_center[int(index[i])]为新加入的聚类中心与第i个点对应的聚类中心
            #之间的距离
            if np.sqrt(center_to_center[int(index[i])])/2.0 > np.sqrt(d[i]):   #注意：这里要有根号才满足
                #第i个点对应的中心仍然为index[i]
                tot = tot + d[i]
                #continue
            else:
                #这里需要先计算下界
                lower_bound = calEuclidean(mean_info[i,:],mean_info[new_add_index,:]) * block_size
                if lower_bound<d[i]:
                    d[i],index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,d[i],index[i])
                    cal_dis_num = cal_dis_num + 1
                    tot = tot + d[i]
                else:
                    tot = tot + d[i]
        tot = tot * rand_[temp_k]   #转动轮盘
        for i, di in enumerate(d):
            tot -= di
            if tot > 0:
                continue
            centers=np.vstack((centers,dataset[i,:]))
            new_add_index = i  #新加入的center的下标
            break
    return centers,cal_dis_num


def LBF_PAA(dataset,k,block_size,rdx,rand_):
    #dataset:数据，n*dim的array
    #k：聚类的个数
    mean_info = preprocessing_with_mean(dataset,block_size)
    n = dataset.shape[0]  
    #rdx = np.random.choice(range(n), 1)
    #rdx = np.array([save_index[0]])  #为了保证两次运行kmeans++结果一致
    new_add_index = rdx[0]
    centers = dataset[rdx]  #初始的
    #centers = np.vstack((centers,dataset[1,:]))
    d = np.array(np.ones(n)*np.inf)  #用于使用轮盘法计算概率，d[i]为第i个点与其最近中心的距离
    index = np.array(np.ones(n)*0)
    cal_dis_num = 0  #用于保存计算欧氏距离的次数
    #cal_dis_num_no_lower_bound = 0#用于保存不用lower——bound需要的计算距离次数
    for temp_k in range(0,k-1):
        tot = 0 #用于轮盘法求概率
        for i,point in enumerate(dataset):
            #判断point_i与新加入的聚类中心的lower_bound，
            #只有当lower_bound<d[i]的时候需要计算真实欧氏距离
            lower_bound = calEuclidean(mean_info[i,:],mean_info[new_add_index,:]) * block_size
            #print(len(mean_info[i,:]))
            if lower_bound<d[i]:
                d[i],index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,d[i],index[i])
                cal_dis_num = cal_dis_num + 1
                tot = tot + d[i]
            else:
                tot = tot + d[i]
        tot = tot * rand_[temp_k]   #转动轮盘
        for i, di in enumerate(d):
            tot -= di
            if tot > 0:
                continue
            centers=np.vstack((centers,dataset[i,:]))
            new_add_index = i  #新加入的center的下标
            break
        #centers=np.vstack((centers,dataset[save_index[temp_k+1],:]))
    return centers,cal_dis_num



def CKM(dataset,k,rdx,rand_):
    #dataset:数据，n*dim的array
    #k：聚类的个数
    n = dataset.shape[0]  
    #rdx = np.random.choice(range(n), 1)  #初始的聚类中心随机选择
    new_add_index = rdx[0]
    centers = dataset[rdx]  #初始的
    #centers = np.vstack((centers,dataset[1,:]))
    #d = np.zeros(n)  #用于使用轮盘法计算概率
    cal_dis_num = 0  #用于保存计算欧氏距离的次数
    current_dis = np.array(np.ones(n)*np.inf)   #每个点到其类中心的距离
    current_index = np.array(np.ones(n)*0)      #每个点对应的类中心
    for temp_k in range(1,k):
        #执行k-1次查找类中心的过程
        tot = 0   #用于D采样
        for i,point in enumerate(dataset):
            #centers[len(centers)-1,:]为新加入的类中心
            current_dis[i],current_index[i]=get_nearest_dist_using_pair_small(point,centers[len(centers)-1,:],len(centers)-1,current_dis[i],current_index[i])
            cal_dis_num = cal_dis_num+1
            #get_nearest_dist_using_pair_small(point,new_center,new_index,current_dis,current_index)
            #cal_dis_num = cal_dis_num + dis_num
            tot = tot + current_dis[i]
        
        tot = tot * rand_[temp_k-1]   #转动轮盘
        for i, di in enumerate(current_dis):
            tot -= di
            if tot > 0:
                continue
            centers=np.vstack((centers,dataset[i,:]))
            break
    return centers,cal_dis_num





