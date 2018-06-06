# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 19:41:31 2017

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
    num_ccn=len(neigh_ij)+num_ccn
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
        index_1=random.choice(G.nodes())
        index_2=random.choice(G.nodes())
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
        index_1=random.choice(G.nodes())
        index_2=random.choice(G.nodes())
        try:
            G[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2)))      
    return test_list,no_list,train_graph
##==================================================
#Gi_karate=ig.Graph.Read_Edgelist("network2_5.txt")#基于这些连边使用igraph创建一个新网络
#Gi_karate=Gi_karate.as_undirected()
##print Gi_karate
#Gn_karate= nx.read_edgelist("network2_5.txt")
#Gn_karate = Gn_karate.to_undirected()
#
####对网络进行社团划分===============================
#comm_list=infomap_comm(Gi_karate)
#L=200
#times=100.0
##原始网络所有连边 
#all_edges=Gn_karate.edges() 
#############区分社团内连边和社团间连边##############
#inside_edges=[]
#for edge in all_edges:
#    for community in comm_list:
#        if (int(edge[0]) in community) and (int(edge[1]) in community):
#            inside_edges.append(edge)
#inside_edges=list(set(inside_edges))
#outside_edges=[i for i in all_edges if i not in inside_edges]
###########################################
##ycn_auc=[]
##ycn_auc_std=[]
##yccn2_auc=[]
##yccn2_auc_std=[]
##
##ycn_pre=[]
##ycn_pre_std=[]
##yccn2_pre=[]
##yccn2_pre_std=[]
##
##for s in ['inside','outside','all']:
##    print s
##    auc_result_cn=[]
##    auc_result_ccn2=[]
##    pre_result_cn=[]
##    pre_result_ccn2=[]
##    
##    for i in range(int(times)):
##        
##        print '%d times'%i
##        
##        if s=='inside':
##            test_list,no_list,train_graph=inside_test_no_list(all_edges,inside_edges,Gn_karate)
##        elif s=='outside':
##            test_list,no_list,train_graph=outside_test_no_list(all_edges,outside_edges,Gn_karate)
##        elif s=='all':
##            test_list,no_list,train_graph=original_test_no_list(Gn_karate)
##        
##        ###############计算auc和precision############
##        auc_cn_real=[]
##        auc_ccn2_real=[]
##        auc_cn_false=[]
##        auc_ccn2_false=[]
##        pre_cn_real=[]
##        pre_ccn2_real=[]
##        pre_cn_false=[]
##        pre_ccn2_false=[]
##        
##        for linkij in test_list:
##            cn=CN(train_graph,linkij)
##            ccn2=CCN2(train_graph,linkij,comm_list)
##            auc_cn_real.append(cn)
##            auc_ccn2_real.append(ccn2)
##            pre_cn_real.append((linkij,cn))
##            pre_ccn2_real.append((linkij,ccn2))
##        for linkij in no_list:
##            cn=CN(train_graph,linkij)
##            ccn2=CCN2(train_graph,linkij,comm_list)
##            auc_cn_false.append(cn)
##            auc_ccn2_false.append(ccn2)
##            pre_cn_false.append((linkij,cn))
##            pre_ccn2_false.append((linkij,ccn2))
##        #===================================================
##        auc_result_cn.append(AUC(auc_cn_real,auc_cn_false))
##        auc_result_ccn2.append(AUC(auc_ccn2_real,auc_ccn2_false))
##        pre_result_cn.append(PRECISION(pre_cn_real,pre_cn_false,L))
##        pre_result_ccn2.append(PRECISION(pre_ccn2_real,pre_ccn2_false,L))
###=============================================================
##    ycn_auc.append(sum(auc_result_cn)/times)
##    ycn_auc_std.append(np.std(auc_result_cn))
##    yccn2_auc.append(sum(auc_result_ccn2)/times)
##    yccn2_auc_std.append(np.std(auc_result_ccn2))
##    
##    ycn_pre.append(sum(pre_result_cn)/times)
##    ycn_pre_std.append(np.std(pre_result_cn))
##    yccn2_pre.append(sum(pre_result_ccn2)/times)
##    yccn2_pre_std.append(np.std(pre_result_ccn2))
########################################
ycn_auc=[0.8574134520276964, 0.5014416419386744, 0.699055390702275]
ycn_auc_std=[0.0058559707115900111, 0.0050345675707955742, 0.0063172136335739124]
yccn2_auc=[0.9854154302670638, 0.5134075173095941, 0.7516493570722058]
yccn2_auc_std=[0.0027662169256159105, 0.010169284451065236, 0.008851796056458968]
ycn_pre=[0.9819499999999998, 0.23310000000000003, 0.9601]
ycn_pre_std=[0.011872131232428331, 0.028946329646433587, 0.022393972403305307]
yccn2_pre=[0.9880999999999998, 0.3599499999999999, 0.9754999999999993]
yccn2_pre_std=[0.0088820042783146723, 0.041752215510078031, 0.013275918047351764]

plt.figure(1,figsize=(7,9))
ax1=plt.subplot(211)
ax2=plt.subplot(212)
plt.subplots_adjust(wspace=0.4)
plt.rcParams['font.size']=15

X=np.arange(3)+1
plt.sca(ax1)
plt.bar(X-0.1,ycn_auc,width = 0.2,facecolor = 'yellowgreen',edgecolor = 'white',
        align="center",label='CN',yerr=ycn_auc_std)
plt.bar(X+0.1,yccn2_auc,width = 0.2,facecolor = 'pink',edgecolor = 'white',
        align="center",label='CCN2',yerr=yccn2_auc_std)
plt.ylim(0.45,1.01)
plt.xticks(X,['inside_edges','outside_edges','all_edges'])
plt.yticks(np.arange(0.45,0.9,0.1))
plt.text(0.2,0.98,'(a)')
plt.xlim(0.5,3.5)
plt.ylabel('AUC')
plt.legend(loc='best')

plt.sca(ax2)
plt.bar(X-0.1,ycn_pre,width = 0.2,facecolor = 'yellowgreen',edgecolor = 'white',
        align="center",yerr=ycn_pre_std)
plt.bar(X+0.1,yccn2_pre,width = 0.2,facecolor = 'pink',edgecolor = 'white',
        align="center",yerr=yccn2_pre_std)

plt.ylim(0.1,1.05)
plt.xticks(X,['inside_edges','outside_edges','all_edges'])
plt.yticks(np.arange(0.1,1,0.2))
plt.text(0.2,1,'(b)')
plt.xlim(0.5,3.5)
plt.ylabel('Precision')
plt.legend(loc='best')
#plt.savefig('OVSI.pdf',bbox_inches='tight')