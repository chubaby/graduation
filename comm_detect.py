# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:33:38 2017

@author: QiYue
"""
import networkx as nx                 #导入networkx网络分析模块，缩写为nx

#==========================================================
#基于K派系过滤的社团检测
def k_comm(Gn):
    comm_list_G=list(nx.k_clique_communities(Gn, 3))   # k=3
    comm_list=[]
    for item in comm_list_G:
        item=map(int,item)
        comm_list.append(list(item))
    return comm_list                                           
#===============================================
#基于fast greedy算法的社团检测
def fastgreedy_comm(Gi):
    h1=Gi.community_fastgreedy(weights=None)   # fastgreedy算法社团检测
    community_list=list(h1.as_clustering())     # 按照默认Q值最大的原则，对系统树图进行切割
    return community_list
#=================================================
## 基于GN算法的社团检测
def GN_comm(Gi):
    h1=Gi.community_edge_betweenness(clusters=None, 
                                            directed=False, weights=None)   # GN算法社团检测
    comm_list=list(h1.as_clustering())        # 按照Q最大的原则对系统树图进行切割，
    return comm_list
##===================================================================
#基于标签传播label propagation的社团检测
def label_pro_comm(Gi):
    comm_list_G=Gi.community_label_propagation()
    comm_list=[]
    for item in comm_list_G:
        comm_list.append(item)
    return comm_list
##==========================================================================
##基于拉普拉斯矩阵的特征向量Leading eigenvector的社团检测
def lead_eigenvector_comm(Gi):
    comm_list_G=Gi.community_leading_eigenvector(clusters=None)
    comm_list=[]
    for item in comm_list_G:
        comm_list.append(item)    
    return comm_list
##===========================================================================