# -*- coding: utf-8 -*-
"""
Created on Sun May 06 18:37:13 2018

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 10:06:22 2017

@author: QiYue
"""
import networkx as nx
import random
import numpy as np
from comm_detect import *
import igraph as ig
import matplotlib.pyplot as plt
import copy
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
#定义precision的程序
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
Gn= nx.read_edgelist("network2_5.txt")
#Gn = Gn.to_undirected()

Gi_karate_d=ig.Graph.Read_Edgelist("network2_5.txt")
Gi_karate=Gi_karate_d.as_undirected()
number_of_edges=ig.Graph.ecount(Gi_karate)

###社团划分===============================
#algorithm=['k_comm','fastgreedy_comm','GN_comm',multilevel_comm，'infomap_comm']
comm_list=k_comm(Gn)   #Gi_karate
#计算Q值
     
L=200   
times=10.0     
auc_result=[]
pre_result=[]
for i in range(int(times)):
    
    #################################test_list and no_list#########
    train_graph=copy.deepcopy(Gn)#拷贝网络
    train_list=train_graph.edges()#获取所有网络连边
    test_list=[]#准备存储测试集连边
    length=int(0.1*len(train_list))
    #划分测试集（10%）和训练集（90%）
    while len(test_list)<length:
        linkij=random.choice(train_list)
        if train_graph.degree(linkij[0])>=1 and train_graph.degree(linkij[1])>=1:
            test_list.append(linkij)
            train_list.remove(linkij)
            train_graph.remove_edge(linkij[0],linkij[1]) 
        
    no_list=[]                           #要预测的  ，不存在的连边，验证集（10%）
    while len(no_list)<length:
        index_1=random.choice(Gn.nodes())
        index_2=random.choice(Gn.nodes())
        try:
            Gn[index_1][index_2]>0             
        except:
            if index_1!=index_2:
                no_list.append((min(index_1,index_2),max(index_1,index_2))) 
    ####################################################
    real2=[]
    false2=[]
    real_2=[]
    false_2=[]
    for linkij in test_list:
        ccn2=CCN2(train_graph,linkij,comm_list)
        real2.append(ccn2)
        real_2.append((linkij,ccn2))
    for linkij in no_list:
        ccn2=CCN2(train_graph,linkij,comm_list)
        false2.append(ccn2)
        false_2.append((linkij,ccn2))
    auc_result.append(AUC(real2,false2))
    pre_result.append(PRECISION(real_2,false_2,L))
##########################
print sum(auc_result)/times#y_auc_ccn2
print np.std(auc_result)#标准差auc_std_ccn2
print sum(pre_result)/times#y_pre_ccn2
print np.std(pre_result)#pre_std_ccn2
#####################draw###################################
##########################画图#############################
#algorithm=['k_comm','fastgreedy_comm','GN_comm','multilevel_comm','infomap_comm']
#y_auc_ccn2=[   ]#5个数。每个数对应一种算法
#auc_std_ccn2=[]
#y_pre_ccn2=[]
#pre_std_ccn2=[]
####################################################################
#plt.figure(1,figsize=(9,13))
#ax1=plt.subplot(211)
#ax2=plt.subplot(212)
#plt.subplots_adjust(hspace=0.2)
#plt.rcParams['font.size']=17
#
#xlist=np.arange(5)
#########################第一个子图######################
#plt.sca(ax1)
#plt.bar(xlist,y_auc_ccn2,width=0.4,facecolor = 'lightskyblue',edgecolor='white',
#    yerr=auc_std_ccn2,align="center")   
#for x,y,z in zip(xlist,algr_Q,y_auc_ccn2):
#    plt.text(x, z+0.01, '%.3f' %y, ha='center', va= 'bottom')    
#plt.ylabel('AUC')
##plt.xlabel('algorithm')
#plt.xticks(xlist,algorithm,size=14)
#plt.yticks(np.arange(0.55,0.8,0.05))
#plt.text(-1,0.771,'(a)')
#plt.xlim(-0.5,7.5)
#plt.ylim(0.55,0.78)
#plt.legend(loc='best')
########################第二个子图#################################
#plt.sca(ax2)
#plt.bar(xlist,y_pre_ccn2,width=0.4,facecolor = 'yellowgreen',edgecolor='white',
#        yerr=pre_std_ccn2,align="center")   
#plt.ylabel('Precision')
#plt.xlabel('algorithm')
#plt.xticks(xlist,algorithm,size=14)
#plt.yticks(np.arange(0.6,1,0.1))
#plt.text(-1,1.01,'(b)')
#plt.ylim(0.58,1.03)
#plt.xlim(-0.5,7.5)
#plt.legend(loc='best')
    
#plt.savefig('ALLDA_Q.pdf',bbox_inches='tight')