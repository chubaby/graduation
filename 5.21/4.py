 # -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 17:46:14 2017

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
#函数ccn2=cn+两个节点邻居在同一社区的数量
def CCN2(G,nodeij,comm_list): 
    node_i=nodeij[0]
    node_j=nodeij[1]
    neigh_i=G.neighbors(node_i)
    neigh_j=G.neighbors(node_j)    
    num_ccn=0
    for nodei in neigh_i:
        for nodej in neigh_j:
            for community in comm_list:
                if (int(nodei) in community) and (int(nodej) in community):
                    num_ccn+=1
    neigh_ij=set(neigh_i)&set(neigh_j)
    num_ccn+=len(neigh_ij)
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
            test_list.append(linkij)
            train_list.remove(linkij)
            train_graph.remove_edge(linkij[0],linkij[1]) 
        
    no_list=[]                           #要预测的  
    while len(no_list)<length:
        index_1=random.choice(G.nodes())
        index_2=random.choice(G.nodes())
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2)))
    ##############计算auc和precision############
    real1=[]
    real2=[]
    false1=[]
    false2=[]
    real_1=[]
    real_2=[]
    false_1=[]
    false_2=[]
    for linkij in test_list:
        cn=CN(train_graph,linkij)
        ccn2=CCN2(train_graph,linkij,comm_list)
        real1.append(cn)
        real2.append(ccn2)
        real_1.append((linkij,cn))
        real_2.append((linkij,ccn2))
    for linkij in no_list:
        cn=CN(train_graph,linkij)
        ccn2=CCN2(train_graph,linkij,comm_list)
        false1.append(cn)
        false2.append(ccn2)
        false_1.append((linkij,cn))
        false_2.append((linkij,ccn2))
    AUC_CN=AUC(real1,false1)
    AUC_CCN2=AUC(real2,false2) 
    PRE_CN=PRECISION(real_1,false_1,L)
    PRE_CCN2=PRECISION(real_2,false_2,L)
    return AUC_CN,AUC_CCN2,PRE_CN,PRE_CCN2
##==================================================
#Gi_karate_d=ig.Graph.Read_Edgelist("network2_5.txt")#基于这些连边使用igraph创建一个新网络
#Gi_karate=Gi_karate_d.as_undirected()
##print Gi_karate
#Gn= nx.read_edgelist("network2_5.txt")
#Gn = Gn.to_undirected()
#
##原始网络所有连边 
#all_edges=Gn.edges()
####对网络进行社团划分===============================
#comm_list=infomap_comm(Gi_karate)
#L=200
########原始网络计算Q###########
##Q_modularity=Q(Gn,Gn,comm_list)
##print 'Q value of original graph is',Q_modularity
############区分社团内连边和社团间连边##############
#inside_edges=[]
#for edge in all_edges:
#    for community in comm_list:
#        if (int(edge[0]) in community) and (int(edge[1]) in community):
#            inside_edges.append(edge)
#inside_edges=list(set(inside_edges))
#outside_edges=[i for i in all_edges if i not in inside_edges]
#########################################
#
###增加社团内连边网络的所有连边计算Q
##all_edges_increase=increase_Q(inside_edges,outside_edges,comm_list)
##increase_G=nx.Graph()
##increase_G.add_edges_from(all_edges_increase)
##Q_modularity=Q(Gn,increase_G,comm_list)
##print 'Q value of increasing Q graph is ',Q_modularity
#
##减少社团內连边网络的所有连边计算Q
#all_edges_decrease=decrease_Q(inside_edges,outside_edges,comm_list)
#decrease_G=nx.Graph()
#decrease_G.add_edges_from(all_edges_decrease)
#Q_modularity=Q(Gn,decrease_G,comm_list)
#print 'Q value of decreasing Q graph is ',Q_modularity
####
######################################
#times=100.0
#auc_result_cn=[]
#auc_result_ccn2=[]
#pre_result_cn=[]
#pre_result_ccn2=[]
#for i in range(int(times)):
#    print 'times=',i
#    a,b,c,d=auc_precision(decrease_G,comm_list)
#    auc_result_cn.append(a)
#    auc_result_ccn2.append(b)
#    pre_result_cn.append(c)
#    pre_result_ccn2.append(d)
############################
#print 'the auc based on cn is',sum(auc_result_cn)/times
#print 'the auc std based on cn is',np.std(auc_result_cn)
#print
#print 'the auc based on ccn2 is',sum(auc_result_ccn2)/times
#print 'the auc std based on ccn is',np.std(auc_result_ccn2)
#print '-------------------------------------------'
#print 'the precision based on cn is',sum(pre_result_cn)/times
#print 'the precision std based on cn is',np.std(pre_result_cn)
#print
#print 'the precision based on ccn2 is',sum(pre_result_ccn2)/times
#print 'the precision std based on ccn2 is',np.std(pre_result_ccn2)
#############################draw#############################
########################## AUC###########
cn_auc=[0.503,0.515,0.535,0.604,0.69,0.694,0.719,0.756,0.81,0.821,
        0.88,0.933]
cn_auc_std=[0.005,0.005,0.005,0.006,0.007,0.005,0.007,0.006,
            0.006,0.006,0.006,0.006]
ccn2_auc=[0.534,0.589,0.622,0.696,0.743,0.749,0.768,0.801,
          0.85,0.867,0.93,0.984]
ccn2_auc_std=[0.013,0.011,0.012,0.011,0.009,0.007,0.01,0.009,0.007,0.007,
              0.006,0.003]

cn_pre=[0.283,0.288,0.342,0.843,0.941,0.945,0.952,0.964,0.966,
        0.966,0.975,0.978]
cn_pre_std=[0.043,0.036,0.027,0.033,0.029,0.021,0.019,0.017,
            0.016,0.017,0.015,0.012]
ccn2_pre=[0.391,0.653,0.813,0.947,0.96,0.962,0.974,0.979,
          0.98,0.981,0.985,0.986]
ccn2_pre_std=[0.042,0.042,0.037,0.016,0.013,0.007,0.014,0.013,
              0.011,0.011,0.009,0.008]

X=[0.049,0.185,0.241,0.358,0.45,0.476,0.5,0.584,0.68,0.723,0.852,0.962]

plt.figure(1,figsize=(9,7))
ax1=plt.subplot(121)
ax2=plt.subplot(122)
plt.subplots_adjust(wspace=0.4)
plt.rcParams['font.size']=17#设置全局字体大小

plt.sca(ax1)
plt.axis([0.04,0.98,0.49,1])
plt.errorbar(X,cn_auc,color = 'blue',marker='x', markersize=7, yerr=cn_auc_std)
plt.errorbar(X,ccn2_auc,color = 'black',marker='o', markersize=5,yerr=ccn2_auc_std)
plt.xlabel('Q')
plt.yticks(np.arange(0.49,1,0.11))
#plt.xticks(X,[0.049,0.185,0.241,0.358,0.45,'0.476(original)',
#              0.5,0.584,0.68,0.723,0.852,0.962],rotation=50)
plt.text(-0.13,0.985,'(a)')
plt.plot([0.476,0.476],[0.49,1],'r--',lw=1)
plt.ylabel('AUC')
#############precision############
plt.sca(ax2)
#plt.figure(figsize=(9,7))
plt.axis([0.04,0.98,0.25,1])
plt.errorbar(X,cn_pre,color = 'blue',marker='x', markersize=7,label='CN',yerr=cn_pre_std)
plt.errorbar(X,ccn2_pre,color = 'black',marker='o', markersize=5,label='CCN2',yerr=ccn2_pre_std)
plt.xlabel('Q')
plt.ylabel('Precision')
plt.yticks(np.arange(0.3,1,0.1))
#plt.xticks(X,[0.049,0.185,0.241,0.358,0.45,'0.476(original)',
#              0.5,0.584,0.68,0.723,0.852,0.962],rotation=60)
plt.text(-0.2,0.975,'(b)')
plt.plot([0.476,0.476],[0.25,1],'r--',lw=1)
plt.legend(loc='best')
plt.savefig('Q.pdf',bbox_inches='tight')