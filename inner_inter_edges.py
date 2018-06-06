# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:46:59 2018

@author: QiYue
"""

import networkx as nx
import random
from comm_detect import *
import igraph as ig
import matplotlib.pyplot as plt
import  numpy as np
import copy

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
###############################
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
###########################################
def original_test_no_list(G):
    train_graph=copy.deepcopy(G)
    train_list=train_graph.edges()
    test_list=[]
    length=int(0.1*len(train_list))
    while len(test_list)<length:
        linkij=random.choice(train_list)
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
                           #test_list添加linkij
                           # train_list移除linkij
                           #train_graph移除连边
        
    no_list=[]                           #要预测的  
    while len(no_list)<length:
        index_1=               #随机选择一个节点
        index_2=               #随机选择一个节点
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2))) 
    return test_list,no_list,train_graph
############################################
def inside_test_no_list(all_edges,inside_edges,G):
    length=int(0.1*len(all_edges))     
    test_list=random.sample(inside_edges,length) 
    train_graph=copy.deepcopy(G)
    for linkij in test_list:
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
            train_graph.remove_edge(linkij[0],linkij[1]) 
            
    no_list=[]    
    while len(no_list)<length:
        index_1=               #随机选择一个节点
        index_2=               #随机选择一个节点
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2)))      
    return test_list,no_list,train_graph
####################################
def outside_test_no_list(all_edges,outside_edges,G):
    length=int(0.1*len(all_edges)) 
    test_list=random.sample(outside_edges,length) 
    train_graph=copy.deepcopy(G)
    for linkij in test_list:
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
            train_graph.remove_edge(linkij[0],linkij[1]) 
            
    no_list=[]    
    while len(no_list)<length:
        index_1=               #随机选择一个节点
        index_2=               #随机选择一个节点
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2)))      
    return test_list,no_list,train_graph
##==================================================
Gi_karate=ig.Graph.Read_Edgelist(  )#基于这些连边使用igraph创建一个新网络
Gi_karate=Gi_karate.as_undirected()
#print Gi_karate
Gn_karate=           #networkx读取网络
Gn_karate = Gn_karate.to_undirected()

###对网络进行社团划分===============================
comm_list=infomap_comm(    )
L=200
times=100.0
#原始网络所有连边 
all_edges= 
############区分社团内连边和社团间连边##############
inside_edges=[]
for edge in all_edges:
    for community in comm_list:
        if (int(edge[0]) in community) and (int(edge[1]) in community):
            inside_edges.append(edge)
inside_edges=list(set(inside_edges))
outside_edges=
##########################################
ycn_auc=[]
ycn_auc_std=[]

ycn_pre=[]
ycn_pre_std=[]

for s in ['inside','outside','all']:
    print s
    auc_result_cn=[]
    pre_result_cn=[]
    for i in range(int(times)):
        
        print '%d times'%i
        
        if s=='inside':
            test_list,no_list,train_graph=inside_test_no_list(    )
        elif s=='outside':
            test_list,no_list,train_graph=outside_test_no_list(     )
        elif s=='all':
            test_list,no_list,train_graph=original_test_no_list(     )
        
        ###############计算auc和precision############
        auc_cn_real=[]       
        auc_cn_false=[]        
        pre_cn_real=[]        
        pre_cn_false=[]
        for linkij in test_list:
            
            
            
            
            
        for linkij in no_list:
            
            
            
            
            
        #===================================================
        auc_result_cn.append(AUC(auc_cn_real,auc_cn_false))
        pre_result_cn.append(PRECISION(pre_cn_real,pre_cn_false,L))
#=============================================================
    ycn_auc.append(sum(auc_result_cn)/times)
    ycn_auc_std.append(np.std(auc_result_cn))

    ycn_pre.append(sum(pre_result_cn)/times)
    ycn_pre_std.append(np.std(pre_result_cn))
####################画图####################
