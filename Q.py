# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:37:05 2018

@author: QiYue
"""
import networkx as nx
import random
from comm_detect import *
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
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
##############计算模块度################
def Q(G,graph, community_list):
    membership=[]       
    for i in range(max(map(int,G.nodes()))+1):
        membership.append(0)    
    # 根据社团划分对membership赋值
    for i in range(0,len(community_list)):
        nodes=map(int, community_list[i])
        for j in nodes:
            membership[j]=i
    e = 0.0
    a_2 = 0.0
    cluster_degree_table = {}
    for vtx, adj in graph.edge.iteritems():
        label = membership[int(vtx)]
        for neighbor in adj.keys():
            if label == membership[int(neighbor)]:
                e += 1
        if label not in cluster_degree_table:
            cluster_degree_table[label] =0
        cluster_degree_table[label] += len(adj)
    e /= 2 * graph.number_of_edges()
    
    for label, cnt in cluster_degree_table.iteritems():
        a = 0.5 * cnt / graph.number_of_edges()
        a_2 += a * a
    
    Q = e - a_2
    return Q
###################增加Q值############################
def increase_Q(inside_edge,outside_edge,comm_list):
    new_edges=[]
    for community in comm_list:
        if len(community)>2:
            times=random.randint(1,len(community))
            for time in range(times):
                node1=unicode(random.choice(community))
                node2=unicode(random.choice(community))
                while node1==node2:
                    node2=random.choice(community)
                if ([node1,node2] not in inside_edge) and ([node2,node1] not in inside_edge):
    #                    print [node1,node2]
                    inside_edge.append((node1,node2))
    new_edges=inside_edge+random.sample(outside_edge,random.randint(1,len(outside_edge))) 
    return new_edges
################减少Q值################
def decrease_Q(inside_edge,outside_edge,comm_list):
    new_edges=[]
    for i in range(len(comm_list)-1):
        j=i+1
        times=random.randint(1,min(len(comm_list[i]),len(comm_list[j])))
        for time in range(times):
            node1=unicode(random.choice(comm_list[i]))
            node2=unicode(random.choice(comm_list[j]))
            if ([node1,node2] not in outside_edge) and ([node2,node1] not in outside_edge):
#                print [node1,node2]
                outside_edge.append((node1,node2))
    new_edges=outside_edge+random.sample(inside_edge,random.randint(1,len(inside_edge)))
    return new_edges
####################################
def auc_precision(G,comm_list):
    #############训练集和测试集#############
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
        index_1=                #随机选择一个节点
        index_2=               #随机选择一个节点
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2)))
    ##############计算auc和precision############
    



    for linkij in test_list:
        
        
        
    for linkij in no_list:
        
        
        
    AUC_CN=AUC(     )
    PRE_CN=PRECISION(      )

    return AUC_CN,PRE_CN
##==================================================
Gi_karate_d=ig.Graph.Read_Edgelist("network2_5.txt")#基于这些连边使用igraph创建一个新网络
Gi_karate=Gi_karate_d.as_undirected()
#print Gi_karate
Gn= 
Gn = Gn.to_undirected()

#原始网络所有连边 
all_edges=             
###对网络进行社团划分===============================
comm_list=infomap_comm(Gi_karate)
L=200
#######原始网络计算Q###########



###########区分社团内连边和社团间连边##############
inside_edges=[]
for edge in all_edges:
    for community in comm_list:
        if (int(edge[0]) in community) and (int(edge[1]) in community):
            inside_edges.append(edge)
inside_edges=list(set(inside_edges))
outside_edges=[i for i in all_edges if i not in inside_edges]
##增加社团内连边网络的所有连边计算Q
all_edges_increase=increase_Q(       )
increase_G=nx.Graph()
increase_G.add_edges_from(all_edges_increase)
Q_modularity=Q(      )
print 'Q value of increasing Q graph is ',Q_modularity

#减少社团內连边网络的所有连边计算Q
all_edges_decrease=decrease_Q(    )
decrease_G=nx.Graph()
decrease_G.add_edges_from(all_edges_decrease)
Q_modularity=Q(      )
print 'Q value of decreasing Q graph is ',Q_modularity
###
#####################################
times=100.0  #实验100次
auc_result_cn=[]
pre_result_cn=[]
for i in range(int(times)):
    print 'times=',i
    a,c=auc_precision(    )
    auc_result_cn.append(a)
    pre_result_cn.append(c)
###########################
print 'the auc based on cn is',sum(auc_result_cn)/times
print '-------------------------------------------'
print 'the precision based on cn is',sum(pre_result_cn)/times
#############################draw#画图############################







###################################