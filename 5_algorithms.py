# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:19:29 2018

@author: QiYue
"""
import networkx as nx
import random
import numpy as np
from comm_detect import *
import igraph as ig
import matplotlib.pyplot as plt
import copy


#########相似性指标############
def CN(G,nodeij):
    node_i=nodeij[0]
    node_j=nodeij[1]
    neigh_ij=set(G.neighbors(node_i))&set(G.neighbors(node_j))
    num_ccn=len(neigh_ij)
    return num_ccn
#定义评价指标AUC
def AUC(real_edges,false_edges):
    AUC_result=0.0
    for i in range(len(real_edges)):
        if real_edges[i]>false_edges[i]:
            AUC_result=AUC_result+1
        elif real_edges[i]==false_edges[i]:
            AUC_result=AUC_result+0.5            
    return AUC_result/len(real_edges)
######定义评价指标AUC######
def PRECISION(real_edges,false_edges,L):
    topL=[]
    cn_real=sorted(real_edges, key=lambda x:x[1],reverse=True)
    cn_false=sorted(false_edges, key=lambda x:x[1],reverse=True)
    i=0
    j=0
    while len(topL)<=L:
        if cn_real[i][1]>cn_false[j][1]:
            topL.append(cn_real[i])
            i=i+1
        elif cn_real[i][1]< cn_false[j][1]:
            topL.append(cn_false[j])
            j=j+1
        else:
            same=[cn_real[i], cn_false[j]]
            a=random.choice(same)
            topL.append(a)
            same.remove(a)
            topL.append(same)
            i=i+1
            j=j+1     
    m=0.0
    for i in range(L):
        if topL[i] in cn_real[0:L-1]:
            m=m+1           
    return m/L   
##==================================================
Gn=    #networkx 读取网络
Gn = Gn.to_undirected()

Gi_karate_d=ig.Graph.Read_Edgelist(   )#网络名
Gi_karate=Gi_karate_d.as_undirected()

##社团划分===============================
algorithm=[      ] #算法名字
comm_list=         #调用社团划分算法  
     
L=200   
times=100.0     
auc_result=[]
pre_result=[]
for i in range(int(times)):
    print '%d time'%i
    #################################test_list and no_list#########
    train_graph=copy.deepcopy(Gn)
    train_list=        # train_graph中的连边集合
    test_list=[]
    length=int(0.1*len(train_list))
    while len(test_list)<length:
        linkij=               #从训练集中随机选择一条连边
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
                        #test_list中加入linkij
                        #train_list中移除linkij
                        #train_graph移除连边
        
    no_list=[]                           #要预测的  
    while len(no_list)<length:
        index_1=           #随机选择一个节点
        index_2=            #随机选择一个节点
        try:
            Gn[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2))) 
    ####################################################
    
    
    
    for linkij in test_list:
         
         
         
         
    for linkij in no_list:
        
         
         
         
         
    auc_result.append(AUC(    ))
    pre_result.append(PRECISION(    ))  
#####################draw#画图显示##################################

