#! /usr/bin/env python
#! -*- coding=utf-8 -*-

'''
从合图文件中提取信息
窗口中心点考虑坐标及矩阵最大值
viterbi 算法采用硬盘Memmap存储
窗口计算前后向及viterbi
回溯不计算路径虚实
'''
import numpy as np  # 版本是 1.26.4 就不会报错
import warnings
from sqlite_master import sql_master
from multiprocessing import Manager,Process,Queue,shared_memory,Value,Lock,Array
from Graph import GRAPH,load_graph,build_graph
warnings.filterwarnings("ignore")
from datetime import datetime
from tqdm import tqdm
import time
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from numba import jit
import sys
from merge_graph import Trace_zip
import sqlite3
import pandas


class Profile_HMM(object):
    def __init__(self,train_DAG_Path,Viterbi_DAG_Path='',partmeter_name='',parameter_path = ''):
        self.commonBaseDict = {"A":0,"T":1,"C":2,"G":3} 
        self.allBaseDict = {"A":0,"T":1,"C":2,"G":3,'R': 4, 'Y': 5, 'M': 6, 'K': 7, 'S': 8, 'W': 9, 'H': 10, 'B': 11, 'V': 12, 'D': 13, 'N': 14}
        self.allBaseDictReverse = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'R', 5: 'Y', 6: 'M', 7: 'K', 8: 'S', 9: 'W', 10: 'H', 11: 'B', 12: 'V', 13: 'D', 14: 'N'}
        self.alignmentBaseDictionary = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}
        self.degenerateBaseDictionary = {'R': [0, 3], 'Y': [2, 1], 'M': [0, 2], 'K': [3, 1], 'S': [3, 2], 'W': [0, 1], 'H': [0, 1, 2], 'B': [3, 1, 2], 'V': [3, 0, 2], 'D': [3, 0, 1], 'N': [0, 1, 2, 3]}
        self.normalBaseCount = len(self.commonBaseDict)
        self.train_DAG_Path = train_DAG_Path
        self.parameterName = partmeter_name

        if Viterbi_DAG_Path =='':
            self.Viterbi_DAG_Path = train_DAG_Path
        else:
            self.Viterbi_DAG_Path = Viterbi_DAG_Path
        
        
        parameterDict=np.load(parameter_path,allow_pickle=True).item()
        for k in ['_mm','_mi','_md','_dm','_dd','_di','_im','_id','_ii','insert_emission','match_emission']:
            parameterDict[k] = np.array(parameterDict[k],dtype=np.float64)

        self.train_times = 0
        self.M2M_array=parameterDict['_mm'][1:-1]
        self.Match_num = len(self.M2M_array)+1
        self.M2I_array=parameterDict['_mi'][1:]
        self.M2D_array=parameterDict['_md'][1:-1]
        self.D2M_array=parameterDict['_dm'][1:-1]
        self.D2D_array=parameterDict['_dd'][1:-1]
        self.D2I_array=parameterDict['_di'][1:]
        self.I2M_array=parameterDict['_im'][:-1]
        self.I2I_array=parameterDict['_ii']
        self.I2D_array=parameterDict['_id'][:-1]
        self.D2E=parameterDict['_dm'][-1]
        self.I2E=parameterDict['_im'][-1]
        self.M2E=parameterDict['_mm'][-1]
        self.Ie_Matrix=parameterDict['insert_emission']
        
        self.Me_Matrix=parameterDict['match_emission']

        self.pi_M=parameterDict['_mm'][0]
        self.pi_D=parameterDict['_md'][0]
        self.pi_I=parameterDict['_mi'][0]
        
        self.Me_Matrix_degenerate_base=[]
        self.Ie_Matrix_degenerate_base=[]
        for i in ['A','T','C','G']:
            self.Me_Matrix_degenerate_base.append(self.Me_Matrix[:,self.commonBaseDict[i]])
            self.Ie_Matrix_degenerate_base.append(self.Ie_Matrix[:,self.commonBaseDict[i]])
        for base in ["R","Y","M","K","S","W","H","B","V","D","N"]:
            degenerate_base = self.degenerateBaseDictionary[base]
            self.Ie_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Ie_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))
            self.Me_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Me_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))

        self.Me_Matrix_degenerate_base = np.array(self.Me_Matrix_degenerate_base)
        self.Ie_Matrix_degenerate_base = np.array(self.Ie_Matrix_degenerate_base)        
    
    @jit(nopython=True)
    def calculate_alpha_values(last_alpha_M, last_alpha_I, last_alpha_D,  Ie, Me, stateRangeStart, stateRangeEnd,D2D_array,I2D_array,M2D_array,D2I_array,I2I_array,M2I_array,D2M_array,I2M_array,M2M_array,arrayLength):
        alpha_M = np.full(arrayLength, np.NINF)
        alpha_I = np.full(arrayLength + 1, np.NINF)
        alpha_D = np.full(arrayLength, np.NINF)
        alpha_I[0] = last_alpha_I[0] + I2I_array[stateRangeStart] + Ie[stateRangeStart]
        alpha_M[0] = last_alpha_I[0] + I2M_array[stateRangeStart] + Me[stateRangeStart]
        alpha_I[1:stateRangeEnd+1] = np.logaddexp(last_alpha_M + M2I_array[stateRangeStart:stateRangeEnd],last_alpha_I[1:] + I2I_array[stateRangeStart+1:stateRangeEnd+1])
        alpha_I[1:stateRangeEnd+1] = np.logaddexp(alpha_I[1:stateRangeEnd+1],last_alpha_D + D2I_array[stateRangeStart:stateRangeEnd])+ Ie[stateRangeStart+1:stateRangeEnd+1]

        alpha_M[1:stateRangeEnd] = np.logaddexp(last_alpha_M[:-1] + M2M_array[stateRangeStart:stateRangeEnd-1],last_alpha_I[1:-1] + I2M_array[stateRangeStart+1:stateRangeEnd])
        alpha_M[1:stateRangeEnd] = np.logaddexp(alpha_M[1:stateRangeEnd],last_alpha_D[:-1] + D2M_array[stateRangeStart:stateRangeEnd-1]) + Me[stateRangeStart+1:stateRangeEnd]


        alpha_D[0] = alpha_I[0] + I2D_array[stateRangeStart]
        tm = np.logaddexp(alpha_M[:-1] + M2D_array[stateRangeStart:stateRangeEnd-1], alpha_I[1:-1] + I2D_array[stateRangeStart+1:stateRangeEnd])
        for i in range(1, arrayLength):
            alpha_D[i] = np.logaddexp(tm[i-1], alpha_D[i-1] + D2D_array[stateRangeStart+i-1])

        return alpha_I, alpha_M, alpha_D
    
    @jit(nopython=True)
    def calculate_beta_values(nextBeta_M, nextBeta_I,beta_D, stateRangeStart,stateRangeEnd, Match_num, D2I_array, D2M_array, D2D_array, I2I_array, I2M_array, I2D_array, M2M_array, M2I_array,M2D_array,arrayLength):
        
        beta_D[-1] = nextBeta_I[-1] + D2I_array[stateRangeEnd-1]
        
        tm = np.logaddexp(nextBeta_M[1:] + D2M_array[stateRangeStart:stateRangeEnd-1], nextBeta_I[1:-1] + D2I_array[stateRangeStart:stateRangeEnd-1])
        for i in range(arrayLength - 2, -1, -1):
            beta_D[i] = np.logaddexp(tm[i], beta_D[i + 1] + D2D_array[i])

        beta_I = np.logaddexp(nextBeta_I + I2I_array[stateRangeStart:stateRangeEnd+1], np.append(nextBeta_M + I2M_array[stateRangeStart:stateRangeEnd], np.NINF))
        beta_I = np.logaddexp(beta_I,np.append(beta_D + I2D_array[stateRangeStart:stateRangeEnd], np.NINF))

        beta_M = np.logaddexp(np.append(nextBeta_M[1:] + M2M_array[stateRangeStart:stateRangeEnd-1], np.NINF), nextBeta_I[1:] + M2I_array[stateRangeStart:stateRangeEnd])
        beta_M = np.logaddexp(beta_M,np.append(beta_D[1:] + M2D_array[stateRangeStart:stateRangeEnd-1], np.NINF))

        return beta_D, beta_I, beta_M, nextBeta_M, nextBeta_I
    
    @jit(nopython=True)
    def get_intersection(content_virus,viruslist,virus_matrix,originalfragmentLength):
        originalfragmentLength-=1
        tuples_setA = set()
        for x in content_virus.T:
            tuples_setA.add((x[0],x[1]))

        tuples_setB = set()
        size_now=0
        for v in viruslist:
            size = v.shape[1]
            virus_matrix[:,size_now:size_now+size]=v
            size_now+=size
        viruslist_T = virus_matrix.T
        condition = viruslist_T[:,1] > originalfragmentLength
        viruslist_T[condition,1] -= np.uint32(1)
        for x in viruslist_T:
            tuples_setB.add((x[0],x[1]))

        nct = tuples_setA&tuples_setB
        if nct:
            return np.array(list(nct)).T
        else:
            return np.empty((0,1),dtype=np.uint32)
    @jit(nopython=True)
    def get_intersection_no_fork(viruslist,virus_matrix,originalfragmentLength):
        originalfragmentLength-=1
        size_now=0
        for v in viruslist:
            size = v.shape[1]
            virus_matrix[:,size_now:size_now+size]=v
            size_now+=size
        condition = virus_matrix[1,:] > originalfragmentLength
        virus_matrix[1,condition] -= np.uint32(1)
        return virus_matrix
        
    
    @jit(nopython=True)
    def update_state(laststate_local, range_length,nowstate, nowstateindex, delta_index_start, decrease_list):
            while nowstate == 1:
                nowstate = laststate_local[nowstate * range_length + nowstateindex - delta_index_start]
                nowstateindex += decrease_list[nowstate]
            return nowstate, nowstateindex

    @jit(nopython=True)
    def calculate_delta_values(stateRangeStart, stateRangeEnd, arrayLength,
                    last_delta_I, last_delta_M, last_delta_D,
                    I2I_array, I2M_array, M2I_array, M2M_array, D2I_array, D2M_array, I2D_array, D2D_array,
                    Ie, Me,M2D_array):
        
        delta_I = np.full(arrayLength+1, np.NINF, dtype=np.float64)
        delta_M = np.full(arrayLength, np.NINF, dtype=np.float64)
        delta_D = np.full(arrayLength, np.NINF, dtype=np.float64)

        delta_I[0] = last_delta_I[0] + I2I_array[stateRangeStart] + Ie[stateRangeStart]
        delta_M[0] = last_delta_I[0] + I2M_array[stateRangeStart] + Me[stateRangeStart]
        I_array = np.vstack((last_delta_M + M2I_array[stateRangeStart:stateRangeEnd],
                                last_delta_D + D2I_array[stateRangeStart:stateRangeEnd],
                                last_delta_I[1:] + I2I_array[stateRangeStart+1:stateRangeEnd+1]))
        M_array = np.vstack((last_delta_M[:-1] + M2M_array[stateRangeStart:stateRangeEnd-1],
                                last_delta_D[:-1] + D2M_array[stateRangeStart:stateRangeEnd-1],
                                last_delta_I[1:-1] + I2M_array[stateRangeStart+1:stateRangeEnd]))
        maxProbOrigin_I = np.full(arrayLength+1, 2)
        maxProbOrigin_M = np.full(arrayLength, 2)
        maxProbOrigin_D = np.full(arrayLength, 2)
        maxProbOrigin_I[1:stateRangeEnd+1] = np.argmax(I_array, axis=0)
        maxProbOrigin_M[1:stateRangeEnd] = np.argmax(M_array, axis=0)

        for i in range(1,arrayLength+1):
            delta_I[i] = I_array[maxProbOrigin_I[i],i-1]
        for i in range(1,arrayLength):
            delta_M[i] = M_array[maxProbOrigin_M[i],i-1]
        delta_I[1:]+=Ie[stateRangeStart+1:stateRangeEnd+1]
        delta_M[1:]+=Me[stateRangeStart+1:stateRangeEnd]

        delta_D[0] = delta_I[0] + I2D_array[stateRangeStart]

        for i in range(1, arrayLength):
            D_arg = np.array([delta_M[i-1] + M2D_array[stateRangeStart+i-1],
                    delta_D[i-1] + D2D_array[stateRangeStart+i-1],
                    delta_I[i] + I2D_array[stateRangeStart+i]])
            
            maxProbOrigin_D[i] = np.argmax(D_arg)
            delta_D[i] = D_arg[maxProbOrigin_D[i]]


        return delta_I, delta_M, delta_D, maxProbOrigin_I, maxProbOrigin_M, maxProbOrigin_D
    
    def forward(self,alpha_Matrix_M,alpha_Matrix_D,alpha_Matrix_I):
        
        def write_alpha_head(indexlist,alpha_head_dict):
            for index in indexlist:
                clique = self.clique_list[index]
                node = clique[0]
                arrayRangeStart,arrayRangeEnd=self.arrayRangeDict[node]
                baseID = self.allBaseDict[self.DAG.nodeList[node][1][-1]]
                alpha_M,alpha_I,alpha_D=alpha_head_dict[baseID]
                
                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]            

                alpha_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = alpha_M[stateRangeStart:stateRangeEnd]
                alpha_Matrix_I[arrayRangeStart:arrayRangeEnd] = alpha_I[stateRangeStart:stateRangeEnd+1]
                alpha_Matrix_D[arrayRangeStart:arrayRangeEnd-1] = alpha_D[stateRangeStart:stateRangeEnd]
                doneList[node]=0
                write_alpha(clique[1:])
                childenCliques=self.DAG.findChildrenClique(index)
                for i in childenCliques:
                    with lock:
                        indegree_dict[i]-=1
                        if indegree_dict[i]==0:
                            q.put(i)

        def init_alpha_head():
            alpha_head_dict={}
            for baseID in range(self.Me_Matrix_degenerate_base.shape[0]):
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                alpha_M=np.full(self.Match_num, np.NINF)
                alpha_I=np.full(self.Match_num+1, np.NINF)
                alpha_D=np.full(self.Match_num, np.NINF)
                alpha_M[0]=self.pi_M+Me[0]
                alpha_I[0]=self.pi_I+Ie[0]
                last_alpha_D = np.full(self.Match_num, np.NINF, dtype=np.float64)
                last_alpha_D[:self.maxrange] = First_alpha_M_D
                alpha_M[1:] = last_alpha_D[:-1]+self.D2M_array+Me[1:]
                alpha_I[1:] = last_alpha_D+self.D2I_array+Ie[1:]
                alpha_D[0] = alpha_I[0]+self.I2D_array[0]
                tm = np.logaddexp(alpha_M[:-1]+self.M2D_array,alpha_I[1:-1]+self.I2D_array[1:])                        
                for i in range(1,self.Match_num):
                    alpha_D[i] = np.logaddexp(tm[i-1],alpha_D[i-1]+self.D2D_array[i-1])
                alpha_head_dict[baseID] = [alpha_M,alpha_I,alpha_D]
            return alpha_head_dict
        
        def write_alpha(nodelist):
            
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            Match_num = self.Match_num

            for node in nodelist:
                arrayRangeStart,arrayRangeEnd=self.arrayRangeDict[node]
                baseID = self.allBaseDict[self.DAG.nodeList[node][1][-1]]
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                partennodes = self.DAG.findParentNodes(node)
                alist = [self.DAG.edgeWeightDict[(lnode, node)] for lnode in partennodes]
                b = np.sum(alist)

                parentNodeWeightList = [
                    [np.log((self.DAG.edgeWeightDict[(lnode, node)] + 0 / len(partennodes)) / (b + 0)), lnode]
                    for lnode in partennodes
                ]
                parentNodeCount = len(partennodes)
                last_alpha_M_list = np.full((parentNodeCount,Match_num), np.NINF)
                last_alpha_I_list = np.full((parentNodeCount,Match_num + 1), np.NINF)
                last_alpha_D_list = np.full((parentNodeCount,Match_num), np.NINF)

                for i, fnode in enumerate(parentNodeWeightList):
                    _arrayRangeStart,_arrayRangeEnd=self.arrayRangeDict[fnode[1]]
                    weight = fnode[0]
                    _stateRangeStart,_stateRangeEnd = self.stateRangeDict[2*fnode[1]:2*fnode[1]+2]
                    last_alpha_M_list[i][_stateRangeStart:_stateRangeEnd] = alpha_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1] + weight
                    last_alpha_I_list[i][_stateRangeStart:_stateRangeEnd+1] = alpha_Matrix_I[_arrayRangeStart:_arrayRangeEnd] + weight
                    last_alpha_D_list[i][_stateRangeStart:_stateRangeEnd] = alpha_Matrix_D[_arrayRangeStart:_arrayRangeEnd-1] + weight

                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                last_alpha_M_list = last_alpha_M_list[:,stateRangeStart:stateRangeEnd]
                last_alpha_I_list = last_alpha_I_list[:,stateRangeStart:stateRangeEnd+1]
                last_alpha_D_list = last_alpha_D_list[:,stateRangeStart:stateRangeEnd]
                arrayLength = stateRangeEnd-stateRangeStart

                last_alpha_M = np.logaddexp.reduce(last_alpha_M_list,axis=0)
                last_alpha_I = np.logaddexp.reduce(last_alpha_I_list,axis=0)
                last_alpha_D = np.logaddexp.reduce(last_alpha_D_list,axis=0)

                alpha_I, alpha_M, alpha_D = Profile_HMM.calculate_alpha_values(last_alpha_M, last_alpha_I, last_alpha_D,  Ie, Me, stateRangeStart, stateRangeEnd,D2D_array,I2D_array,M2D_array,D2I_array,I2I_array,M2I_array,D2M_array,I2M_array,M2M_array,arrayLength)
                alpha_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = alpha_M
                alpha_Matrix_I[arrayRangeStart:arrayRangeEnd] = alpha_I
                alpha_Matrix_D[arrayRangeStart:arrayRangeEnd-1] = alpha_D
                doneList[node]=0
        
        
        def calculate_alpha(lock,nodelist,alpha_head_dict):
            clique_list = self.clique_list
            write_alpha_head(nodelist,alpha_head_dict)
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1)
                except:
                    with pool_lock:
                        working_bots.value-=1
                    t=0
                    while True:
                        time.sleep(0.1)
                        if  q.qsize()>working_bots.value*2 or not goon_flag.value:
                            t+=1
                        else:
                            t=0
                        if t > 5:
                            with pool_lock:
                                working_bots.value+=1
                                break
                    continue

                while start_node!=None:
                    
                    clique = clique_list[start_node]
                    write_alpha(clique)
                    checklist=self.DAG.findChildrenClique(start_node)
                    todolist=[]
                    for i in checklist:
                        with lock:
                            indegree_dict[i]-=1
                            if indegree_dict[i]==0:
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()
                        
                        for i in todolist:

                            q.put(i)

                    else:
                        start_node=None



        def is_end(goon_flag):

            start = time.time()
            while True:
                time.sleep(1)
                runtime = time.time() - start
                percent = np.round((self.DAG.totalNodes-np.count_nonzero(doneList))/self.DAG.totalNodes,5)
                bar = ('#' * int(percent * 20)).ljust(20)
                mins, secs = divmod(runtime, 60)
                time_format = '{:02d}:{:02d}'.format(int(mins), int(secs))
                sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}% 完成 (运行时间: {time_format} 分钟)')
                sys.stdout.flush()

                if 0==np.count_nonzero(doneList>0):
                    goon_flag.value=0
                    
                    break
        
        lock = Lock()
        goon_flag = Value('i',1)
        arrayRangeStart=self.arrayRangeDict[-1][0]
        alpha_Matrix_D[arrayRangeStart] = self.pi_D
        for i in range(1,self.maxrange):
            alpha_Matrix_D[arrayRangeStart+i] = alpha_Matrix_D[arrayRangeStart+i-1]+self.D2D_array[i-1]
        First_alpha_M_D = alpha_Matrix_D[arrayRangeStart:arrayRangeStart+self.maxrange]


        pool_lock=Lock()
        working_bots = Value('i',self.pool_num)
        
        q = Queue()
                

        alpha_head_dict=init_alpha_head()
        startnodelist = list(self.DAG.startnodeset_clique)
            
        indegree_dict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint16).itemsize)
        indegree_dict =np.ndarray((self.DAG.totalNodes,), dtype=np.int16, buffer=indegree_dict_shm.buf)

        link_keys=self.clique_link
        for link in link_keys:
            indegree_dict[link[1]]+=1
        doneList_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint8).itemsize)
        doneList =np.ndarray((self.DAG.totalNodes,), dtype=np.int8, buffer=doneList_shm.buf)
        doneList[:]=[1]*self.DAG.totalNodes
        
        processlist=[]
        pool_num = self.pool_num
        for idx in range(pool_num):
            processlist.append(Process(target=calculate_alpha, args=(lock,startnodelist[idx::pool_num],alpha_head_dict, )))
        processlist.append(Process(target=is_end, args=(goon_flag,  )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        last_alpha_M_list = []
        last_alpha_I_list = []
        last_alpha_D_list = []
        for fnode in self.graphEndNodes:
            _arrayRangeStart,_arrayRangeEnd=self.arrayRangeDict[fnode]
            weight = np.log(self.DAG.nodeList[fnode][2]/self.sequenceNum)
            _stateRangeStart = self.stateRangeDict[2*fnode]
            _stateRangeEnd = self.stateRangeDict[2*fnode+1]
            temp_alpha_M=np.full(self.Match_num, np.NINF)
            temp_alpha_I=np.full(self.Match_num+1, np.NINF)
            temp_alpha_D=np.full(self.Match_num, np.NINF)
            temp_alpha_M[_stateRangeStart:_stateRangeEnd] = alpha_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1]
            temp_alpha_I[_stateRangeStart:_stateRangeEnd+1] = alpha_Matrix_I[_arrayRangeStart:_arrayRangeEnd]
            temp_alpha_D[_stateRangeStart:_stateRangeEnd] = alpha_Matrix_D[_arrayRangeStart:_arrayRangeEnd-1]
            last_alpha_M_list.append(temp_alpha_M + weight)
            last_alpha_I_list.append(temp_alpha_I + weight)
            last_alpha_D_list.append(temp_alpha_D + weight) 
        
        last_alpha_M = np.logaddexp.reduce(last_alpha_M_list,axis=0)
        last_alpha_I = np.logaddexp.reduce(last_alpha_I_list,axis=0)
        last_alpha_D = np.logaddexp.reduce(last_alpha_D_list,axis=0)

        prob = np.logaddexp.reduce([last_alpha_M[-1]+self.M2E,last_alpha_I[-1]+self.I2E,last_alpha_D[-1]+self.D2E],axis=0)
        print(prob)
        indegree_dict_shm.close()
        indegree_dict_shm.unlink()
        doneList_shm.close()
        doneList_shm.unlink()
        
    def backward(self,beta_Matrix_M,beta_Matrix_D,beta_Matrix_I,left_beta_Matrix_M,left_beta_Matrix_I,problist):
        def write_beta_head(indexlist,beta_head):
            for index in indexlist:
                clique = self.clique_list[index]
                node = clique.pop()
                arrayRangeStart,arrayRangeEnd=self.arrayRangeDict[node]
                weight = np.log(self.DAG.nodeList[node][2]/self.sequenceNum)
                beta_M,beta_I,beta_D = beta_head
                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                arrayLength = stateRangeEnd-stateRangeStart
                nextBeta_M = np.full(arrayLength,np.NINF)
                nextBeta_I = np.full(arrayLength+1,np.NINF)
                beta_M = beta_M[stateRangeStart:stateRangeEnd]+weight
                beta_I = beta_I[stateRangeStart:stateRangeEnd+1]+weight
                beta_D = beta_D[stateRangeStart:stateRangeEnd]+weight
                beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1]=beta_M # [stateRangeStart:stateRangeEnd]
                beta_Matrix_I[arrayRangeStart:arrayRangeEnd]=beta_I # [stateRangeStart:stateRangeEnd+1]
                beta_Matrix_D[arrayRangeStart:arrayRangeEnd-1]=beta_D # [stateRangeStart:stateRangeEnd]

                left_beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = nextBeta_M
                left_beta_Matrix_I[arrayRangeStart:arrayRangeEnd] = nextBeta_I
                doneList[node]=0
                write_beta(clique)
                checklist=self.DAG.findParentClique(index)
                for i in checklist:
                    with lock:
                        outdegreeDict[i]-=1
                        if outdegreeDict[i]==0:
                            q.put(i)
        def init_beta_head():
            Match_num = self.Match_num
            beta_M = np.full(Match_num, np.NINF)
            beta_I = np.full(Match_num + 1, np.NINF)
            beta_D = np.full(Match_num, np.NINF)
            beta_M[-1] = self.M2E 
            beta_I[-1] = self.I2E 
            beta_D[-1] = self.D2E 
            for i in range(Match_num-1)[::-1]:
                beta_D[i] = beta_D[i+1]+self.D2D_array[i]
            beta_M[:-1] = beta_D[1:]+self.M2D_array
            beta_I[:-1] = beta_D+self.I2D_array
            beta_head = [beta_M,beta_I,beta_D]
            return beta_head
        def write_beta(nodes):
            Match_num = self.Match_num
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array

            for node in nodes[::-1]:
                sonnodes = self.DAG.findChildNodes(node)
                son_nodelist = [
                    [np.log(self.DAG.edgeWeightDict[(node, lnode)] / np.sum([self.DAG.edgeWeightDict[(fnode, lnode)] for fnode in self.DAG.findParentNodes(lnode)])), lnode]
                    for lnode in sonnodes
                ]
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                
                
                sonnde_num = len(son_nodelist)
                nextBeta_M_List = np.full((sonnde_num, Match_num),np.NINF, dtype=np.float64)
                nextBeta_I_List = np.full((sonnde_num, Match_num + 1),np.NINF, dtype=np.float64)

                for i, fnode in enumerate(son_nodelist):
                    _arrayRangeStart,_arrayRangeEnd = self.arrayRangeDict[fnode[1]]
                    weight = fnode[0]
                    fathernode_leftlimit,fathernode_rightlimit = self.stateRangeDict[2*fnode[1]:2*fnode[1]+2]
                    # fllll=fathernode_rightlimit-fathernode_leftlimit
                    lo = self.allBaseDict[self.DAG.nodeList[fnode[1]][1]] 
                    nextBeta_M_List[i][fathernode_leftlimit:fathernode_rightlimit] = beta_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1]+self.Me_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit] + weight
                    nextBeta_I_List[i][fathernode_leftlimit:fathernode_rightlimit+1] = beta_Matrix_I[_arrayRangeStart:_arrayRangeEnd]+self.Ie_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit+1] + weight
                
                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                nextBeta_M_List = nextBeta_M_List[:,stateRangeStart:stateRangeEnd]
                nextBeta_I_List = nextBeta_I_List[:,stateRangeStart:stateRangeEnd+1]
                nextBeta_M = np.logaddexp.reduce(nextBeta_M_List, axis=0)
                nextBeta_I = np.logaddexp.reduce(nextBeta_I_List, axis=0)
                arrayLength = stateRangeEnd-stateRangeStart
                beta_D = np.full(arrayLength,np.NINF)
                beta_D, beta_I, beta_M, nextBeta_M, nextBeta_I = Profile_HMM.calculate_beta_values(nextBeta_M, nextBeta_I,beta_D, stateRangeStart,stateRangeEnd, Match_num, D2I_array, D2M_array, D2D_array, I2I_array, I2M_array, I2D_array, M2M_array, M2I_array,M2D_array,arrayLength)

                beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1]=beta_M 
                beta_Matrix_I[arrayRangeStart:arrayRangeEnd]=beta_I
                beta_Matrix_D[arrayRangeStart:arrayRangeEnd-1]=beta_D 

                left_beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = nextBeta_M
                left_beta_Matrix_I[arrayRangeStart:arrayRangeEnd] = nextBeta_I
                doneList[node]=0

        
        def calculate_beta(lock,nodelist,beta_head,):
            clique_list = self.clique_list
            write_beta_head(nodelist,beta_head)
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1)
                except:
                    with pool_lock:
                        working_bots.value-=1
                    t=0
                    while True:
                        time.sleep(0.1)
                        if  q.qsize()>working_bots.value*2 or not goon_flag.value:
                            t+=1
                        else:
                            t=0
                        if t > 5:
                            with pool_lock:
                                working_bots.value+=1
                                break
                    continue

                while start_node!=None:
                    clique = clique_list[start_node]

                    write_beta(clique)
                    checklist=self.DAG.findParentClique(start_node)

                    todolist=[]
                    for i in checklist:
                        with lock:
                            outdegreeDict[i]-=1

                            if outdegreeDict[i]==0:
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()
                        for i in todolist:
                            q.put(i)
                    else:
                        start_node=None

        def is_end(goon_flag):
            while True:
                if 0==np.count_nonzero(doneList>0):
                    goon_flag.value=0
                    break


        outdegreeDict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.clique_num*np.dtype(np.uint16).itemsize)
        outdegreeDict =np.ndarray((self.DAG.clique_num,), dtype=np.int16, buffer=outdegreeDict_shm.buf)
        link_keys = self.clique_link
        for link in link_keys:
            outdegreeDict[link[0]]+=1

        doneList_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint8).itemsize)
        doneList =np.ndarray((self.DAG.totalNodes,), dtype=np.int8, buffer=doneList_shm.buf)
        doneList[:]=[1]*self.DAG.totalNodes

        lock = Lock()
        goon_flag=Value('i',1)
        q = Queue()
        beta_head = init_beta_head()

        endnodelist = list(self.DAG.endnodeset_clique)

        pool_lock=Lock()
        working_bots = Value('i',self.pool_num)
        processlist=[]
        pool_num = self.pool_num
        for idx in range(pool_num):
            processlist.append(Process(target=calculate_beta,args=(lock,endnodelist[idx::pool_num],beta_head, )))
        processlist.append(Process(target=is_end,args=(goon_flag, )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]




        nextBeta_M_List = np.full((len(self.graphStartNodes), self.Match_num),np.NINF, dtype=np.float64)
        nextBeta_I_List = np.full((len(self.graphStartNodes), self.Match_num + 1),np.NINF, dtype=np.float64)
        maxright=0
        graphStartNodes = list(self.graphStartNodes)
        for i, fnode in enumerate(graphStartNodes):
            _arrayRangeStart,_arrayRangeEnd = self.arrayRangeDict[fnode]
            fathernode_leftlimit = self.stateRangeDict[2*fnode]
            fathernode_rightlimit = self.stateRangeDict[2*fnode+1]
            lo = self.allBaseDict[self.DAG.nodeList[fnode][1]]

            weight=0
            if fathernode_rightlimit>maxright:
                maxright=fathernode_rightlimit

            nextBeta_M_List[i][fathernode_leftlimit:fathernode_rightlimit] = beta_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1]+self.Me_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit] + weight
            nextBeta_I_List[i][fathernode_leftlimit:fathernode_rightlimit+1] = beta_Matrix_I[_arrayRangeStart:_arrayRangeEnd]+self.Ie_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit+1] + weight
        

        stateRangeStart = 0
        stateRangeEnd = self.maxrange
        arrayLength = stateRangeEnd-stateRangeStart
        beta_D = np.full(arrayLength,np.NINF)
        nextBeta_M_List = nextBeta_M_List[:,stateRangeStart:stateRangeEnd]

        nextBeta_I_List = nextBeta_I_List[:,stateRangeStart:stateRangeEnd+1]
        nextBeta_M = np.logaddexp.reduce(nextBeta_M_List, axis=0)
        nextBeta_I = np.logaddexp.reduce(nextBeta_I_List, axis=0)
        beta_D, beta_I, beta_M, nextBeta_M, nextBeta_I = Profile_HMM.calculate_beta_values(nextBeta_M, nextBeta_I,beta_D, stateRangeStart,stateRangeEnd, self.Match_num, self.D2I_array, self.D2M_array, self.D2D_array, self.I2I_array, self.I2M_array, self.I2D_array, self.M2M_array, self.M2I_array,self.M2D_array,arrayLength)

        arrayRangeStart = self.arrayRangeDict[-1][0]

        beta_Matrix_M[arrayRangeStart:arrayRangeStart+arrayLength]=beta_M
        beta_Matrix_I[arrayRangeStart:arrayRangeStart+arrayLength+1]=beta_I
        beta_Matrix_D[arrayRangeStart:arrayRangeStart+arrayLength]=beta_D

        left_beta_Matrix_M[arrayRangeStart:arrayRangeStart+arrayLength] = nextBeta_M
        left_beta_Matrix_I[arrayRangeStart:arrayRangeStart+arrayLength+1] = nextBeta_I
        
        prob = np.logaddexp.reduce([nextBeta_M[0]+self.pi_M,nextBeta_I[0]+self.pi_I,beta_D[0]+self.pi_D],axis=0)
        outdegreeDict_shm.close()
        outdegreeDict_shm.unlink()
        doneList_shm.close()
        doneList_shm.unlink()
        problist.append(prob)

        
    def estep(self,alpha_Matrix_M,alpha_Matrix_I,alpha_Matrix_D,beta_Matrix_M,beta_Matrix_D,beta_Matrix_I,left_beta_Matrix_M,left_beta_Matrix_I):

        def calculate_gamma(nodes,gamma_o_M_list,gamma_o_I_list,E_MM_list,E_MD_list,E_MI_list,E_II_list,E_IM_list,E_DM_list,E_DD_list,MEi_list,IEi_list,DEi_list):
            Match_num = self.Match_num
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            degenerate_base_dict = self.degenerateBaseDictionary.copy()
            degenerate_base_dict['A']=[0]
            degenerate_base_dict['T']=[1]
            degenerate_base_dict['C']=[2]
            degenerate_base_dict['G']=[3]
            allBaseDictReverse = self.allBaseDictReverse
            gamma_o_M = np.log(np.zeros((Match_num, self.normalBaseCount),dtype=np.float64))
            gamma_o_I = np.log(np.zeros((Match_num+1, self.normalBaseCount),dtype=np.float64))
            E_MM = np.full(Match_num-1, np.NINF)
            E_MD = np.full(Match_num-1, np.NINF)
            E_MI = np.full(Match_num, np.NINF)
            E_II = np.full(Match_num+1, np.NINF)
            E_IM = np.full(Match_num, np.NINF)
            E_ID = np.full(Match_num, np.NINF)
            E_DM = np.full(Match_num-1, np.NINF)
            E_DD = np.full(Match_num-1, np.NINF)
            E_DI = np.full(Match_num, np.NINF)
            MEi = np.full(Match_num, np.NINF)

            IEi = np.full(Match_num+1, np.NINF)
            DEi = np.full(Match_num, np.NINF)
            nodes = set(nodes)
            stnodes = self.graphStartNodes|set([-1])
            spnodes = nodes&stnodes
            nodes -=spnodes
            for node in spnodes:
                stateRangeStart=0
                if node!=-1:
                    stateRangeEnd = self.stateRangeDict[2*node+1]
                else:
                    stateRangeEnd = self.maxrange

                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict[self.DAG.nodeList[node][1]]
                alpha_M = alpha_Matrix_M[arrayRangeStart:arrayRangeEnd-1]
                alpha_I = alpha_Matrix_I[arrayRangeStart:arrayRangeEnd]
                alpha_D = alpha_Matrix_D[arrayRangeStart:arrayRangeEnd-1]
                beta_M = beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1]
                beta_I = beta_Matrix_I[arrayRangeStart:arrayRangeEnd]
                beta_D = beta_Matrix_D[arrayRangeStart:arrayRangeEnd-1]
                nextBeta_M = left_beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1]
                nextBeta_I = left_beta_Matrix_I[arrayRangeStart:arrayRangeEnd]
                gamma_M =  alpha_M+beta_M 
                gamma_I =  alpha_I+beta_I 
                gamma_D =  alpha_D+beta_D 
                E_MM[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_M[:-1]+M2M_array[stateRangeStart:stateRangeEnd-1]+nextBeta_M[1:],E_MM[stateRangeStart:stateRangeEnd-1])
                E_MD[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_M[:-1]+M2D_array[stateRangeStart:stateRangeEnd-1]+beta_D[1:],E_MD[stateRangeStart:stateRangeEnd-1])
                E_MI[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_M+M2I_array[stateRangeStart:stateRangeEnd]+nextBeta_I[1:],E_MI[stateRangeStart:stateRangeEnd])

                E_II[stateRangeStart:stateRangeEnd+1] = np.logaddexp(alpha_I+I2I_array[stateRangeStart:stateRangeEnd+1]+nextBeta_I,E_II[stateRangeStart:stateRangeEnd+1])
                E_IM[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_I[:-1]+I2M_array[stateRangeStart:stateRangeEnd]+nextBeta_M,E_IM[stateRangeStart:stateRangeEnd])
                E_ID[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_I[:-1]+I2D_array[stateRangeStart:stateRangeEnd]+beta_D,E_ID[stateRangeStart:stateRangeEnd])

                E_DM[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_D[:-1]+D2M_array[stateRangeStart:stateRangeEnd-1]+nextBeta_M[1:],E_DM[stateRangeStart:stateRangeEnd-1])
                E_DD[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_D[:-1]+D2D_array[stateRangeStart:stateRangeEnd-1]+beta_D[1:],E_DD[stateRangeStart:stateRangeEnd-1])
                E_DI[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_D+D2I_array[stateRangeStart:stateRangeEnd]+nextBeta_I[1:],E_DI[stateRangeStart:stateRangeEnd])
                
                MEi[stateRangeStart:stateRangeEnd] = np.logaddexp(gamma_M,MEi[stateRangeStart:stateRangeEnd])
                IEi[stateRangeStart:stateRangeEnd+1] = np.logaddexp(gamma_I,IEi[stateRangeStart:stateRangeEnd+1])
                DEi[stateRangeStart:stateRangeEnd] = np.logaddexp(gamma_D,DEi[stateRangeStart:stateRangeEnd])
                self.gamma_M[node] = [gamma_M[0]]
                self.gamma_I[node] = [gamma_I[0]]
                self.gamma_D[node] = [gamma_D[0]]

            for node in nodes:

                stateRangeStart = self.stateRangeDict[2*node]
                stateRangeEnd = self.stateRangeDict[2*node+1]

                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict[self.DAG.nodeList[node][1]]
                alpha_M = alpha_Matrix_M[arrayRangeStart:arrayRangeEnd-1]
                alpha_I = alpha_Matrix_I[arrayRangeStart:arrayRangeEnd]
                alpha_D = alpha_Matrix_D[arrayRangeStart:arrayRangeEnd-1]
                beta_M = beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1]
                beta_I = beta_Matrix_I[arrayRangeStart:arrayRangeEnd]
                beta_D = beta_Matrix_D[arrayRangeStart:arrayRangeEnd-1]
                nextBeta_M = left_beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1]
                nextBeta_I = left_beta_Matrix_I[arrayRangeStart:arrayRangeEnd]
                gamma_M =  alpha_M+beta_M 
                gamma_I =  alpha_I+beta_I 
                gamma_D =  alpha_D+beta_D 
                
                bases = degenerate_base_dict[allBaseDictReverse[baseID]]

                
                E_MM[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_M[:-1]+M2M_array[stateRangeStart:stateRangeEnd-1]+nextBeta_M[1:],E_MM[stateRangeStart:stateRangeEnd-1])
                E_MD[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_M[:-1]+M2D_array[stateRangeStart:stateRangeEnd-1]+beta_D[1:],E_MD[stateRangeStart:stateRangeEnd-1])
                E_MI[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_M+M2I_array[stateRangeStart:stateRangeEnd]+nextBeta_I[1:],E_MI[stateRangeStart:stateRangeEnd])

                E_II[stateRangeStart:stateRangeEnd+1] = np.logaddexp(alpha_I+I2I_array[stateRangeStart:stateRangeEnd+1]+nextBeta_I,E_II[stateRangeStart:stateRangeEnd+1])
                E_IM[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_I[:-1]+I2M_array[stateRangeStart:stateRangeEnd]+nextBeta_M,E_IM[stateRangeStart:stateRangeEnd])
                E_ID[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_I[:-1]+I2D_array[stateRangeStart:stateRangeEnd]+beta_D,E_ID[stateRangeStart:stateRangeEnd])

                E_DM[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_D[:-1]+D2M_array[stateRangeStart:stateRangeEnd-1]+nextBeta_M[1:],E_DM[stateRangeStart:stateRangeEnd-1])
                E_DD[stateRangeStart:stateRangeEnd-1] = np.logaddexp(alpha_D[:-1]+D2D_array[stateRangeStart:stateRangeEnd-1]+beta_D[1:],E_DD[stateRangeStart:stateRangeEnd-1])
                E_DI[stateRangeStart:stateRangeEnd] = np.logaddexp(alpha_D+D2I_array[stateRangeStart:stateRangeEnd]+nextBeta_I[1:],E_DI[stateRangeStart:stateRangeEnd])
                
                MEi[stateRangeStart:stateRangeEnd] = np.logaddexp(gamma_M,MEi[stateRangeStart:stateRangeEnd])
                IEi[stateRangeStart:stateRangeEnd+1] = np.logaddexp(gamma_I,IEi[stateRangeStart:stateRangeEnd+1])
                DEi[stateRangeStart:stateRangeEnd] = np.logaddexp(gamma_D,DEi[stateRangeStart:stateRangeEnd])

                for base in bases:
                    gamma_o_M[stateRangeStart:stateRangeEnd,base] = np.logaddexp(gamma_M-np.log(len(bases)),gamma_o_M[stateRangeStart:stateRangeEnd,base])
                    gamma_o_I[stateRangeStart:stateRangeEnd+1,base] = np.logaddexp(gamma_I-np.log(len(bases)),gamma_o_I[stateRangeStart:stateRangeEnd+1,base])


            gamma_o_M_list.append(gamma_o_M)
            gamma_o_I_list.append(gamma_o_I)
            E_MM_list.append(E_MM)
            E_MD_list.append(E_MD)
            E_MI_list.append(E_MI)
            E_II_list.append(E_II)
            E_IM_list.append(E_IM)
            E_ID_list.append(E_ID)
            E_DM_list.append(E_DM)
            E_DD_list.append(E_DD)
            E_DI_list.append(E_DI)
            MEi_list.append(MEi)
            DEi_list.append(DEi)
            IEi_list.append(IEi)



        problist = Manager().list()
        
        problist.append([])
        starttime = datetime.now()
        allprocesslist=[]

        allprocesslist.append(Process(target=self.forward,args=(alpha_Matrix_M,alpha_Matrix_D,alpha_Matrix_I, )))
        allprocesslist.append(Process(target=self.backward,args=(beta_Matrix_M,beta_Matrix_D,beta_Matrix_I,left_beta_Matrix_M,left_beta_Matrix_I,problist, )))
        [p.start() for p in allprocesslist]
        [p.join() for p in allprocesslist]
        prob = problist[-1]
        print()
        print('Prob',prob)
        

        self.gamma_M = Manager().dict()
        self.gamma_I = Manager().dict()
        self.gamma_D = Manager().dict()

        self.nodelist = list(range(self.DAG.totalNodes))
        self.nodelist.append(-1)
        pool_num = min(self.pool_num*3,80)

        gamma_o_M_list = Manager().list()
        gamma_o_I_list = Manager().list()
        
        E_MM_list = Manager().list()
        E_MD_list = Manager().list()
        E_MI_list = Manager().list()
        E_II_list = Manager().list()
        E_IM_list = Manager().list()
        E_ID_list = Manager().list()
        E_DM_list = Manager().list()
        E_DD_list = Manager().list()
        E_DI_list = Manager().list()
        MEi_list = Manager().list()
        IEi_list = Manager().list()
        DEi_list = Manager().list()

        
        processlist =[]
        for index in range(pool_num):
            processlist.append(Process(target=calculate_gamma,args=(self.nodelist[index::pool_num],gamma_o_M_list,gamma_o_I_list,E_MM_list,E_MD_list,E_MI_list,E_II_list,E_IM_list,E_DM_list,E_DD_list,MEi_list,IEi_list,DEi_list, )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]

        self.gamma_o_M = np.logaddexp.reduce(gamma_o_M_list,axis=0) - prob
        self.gamma_o_I = np.logaddexp.reduce(gamma_o_I_list,axis=0) - prob
        self.E_MM = np.logaddexp.reduce(E_MM_list,axis=0)-prob
        self.E_MD = np.logaddexp.reduce(E_MD_list,axis=0)-prob
        self.E_MI = np.logaddexp.reduce(E_MI_list,axis=0)-prob
        self.E_II = np.logaddexp.reduce(E_II_list,axis=0)-prob
        self.E_IM = np.logaddexp.reduce(E_IM_list,axis=0)-prob
        self.E_ID = np.logaddexp.reduce(E_ID_list,axis=0)-prob
        self.E_DM = np.logaddexp.reduce(E_DM_list,axis=0)-prob
        self.E_DD = np.logaddexp.reduce(E_DD_list,axis=0)-prob
        self.E_DI = np.logaddexp.reduce(E_DI_list,axis=0)-prob
        self.MEi  = np.logaddexp.reduce(MEi_list,axis=0) - prob
        self.DEi  = np.logaddexp.reduce(DEi_list,axis=0) - prob
        self.IEi  = np.logaddexp.reduce(IEi_list,axis=0) - prob
        
        return prob,starttime
    
    def mstep(self,starttime):
        self.Ie_Matrix = self.gamma_o_I-self.IEi.reshape(-1, 1)
        self.Me_Matrix = self.gamma_o_M-self.MEi.reshape(-1, 1)
        adds = np.array([self.emProbAdds_Match]*(self.normalBaseCount),dtype=np.float64)
        head_adds = np.array([self.emProbAdds_Match_head]*(self.normalBaseCount),dtype=np.float64)
        tail_adds = np.array([self.emProbAdds_Match_tail]*(self.normalBaseCount),dtype=np.float64)
        head_length = self.head_length
        tail_length = self.tail_length
        for i in range(self.Match_num):
            if i<=head_length:
                Madds = head_adds
            elif i>=self.Match_num-tail_length:
                Madds = tail_adds
            else:
                Madds = adds
            tmp = np.logaddexp(self.Me_Matrix[i],Madds)
            Psum = np.logaddexp.reduce(tmp,axis=0)
            self.Me_Matrix[i] = tmp-Psum

        self.Ie_Matrix = np.full_like(self.Ie_Matrix,np.log(1/4))
        Adds_m_ = [self.trProbAdds_mm,self.trProbAdds_mi,self.trProbAdds_md]
        
        self.M2M_array = self.E_MM-self.MEi[:-1]
        self.M2I_array = self.E_MI-self.MEi
        self.M2D_array = self.E_MD-self.MEi[:-1]
        for i in range(self.Match_num-1):
            Adds = Adds_m_
            tmp = np.zeros(3,dtype=np.float64)
            tmp[0] = np.logaddexp(self.M2M_array[i],Adds[0])
            tmp[1] = np.logaddexp(self.M2I_array[i],Adds[1])
            tmp[2] = np.logaddexp(self.M2D_array[i],Adds[2])
            Psum = np.logaddexp.reduce(tmp,axis=0)
            self.M2M_array[i] = tmp[0] - Psum
            self.M2I_array[i] = tmp[1] - Psum
            self.M2D_array[i] = tmp[2] - Psum
        tmp = np.zeros(2,dtype=np.float64)
        tmp[0] = np.logaddexp(self.M2E,self.trProbAdds_mend)
        tmp[1] = np.logaddexp(self.M2I_array[-1],self.trProbAdds_mi_tail)
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.M2E = tmp[0]- Psum
        self.M2I_array[-1] = tmp[1]- Psum

        Adds_im =  self.trProbAdds_im
        Adds_ii =  self.trProbAdds_ii
        Adds_IE =  self.trProbAdds_iend # np.log(0.01)
        Adds_iitail = self.trProbAdds_ii_tail # np.log(0.99)

        self.I2D_array = self.E_ID-self.IEi[:-1]
        self.I2I_array[1:-1] = self.E_II[1:-1]-self.IEi[1:-1]
        self.I2M_array[1:] = self.E_IM[1:]-self.IEi[1:-1]

        i=0
        tmp = np.zeros(2,dtype=np.float64)
        tmp[0] = np.logaddexp(self.I2M_array[i],Adds_im)
        tmp[1] = np.logaddexp(self.I2I_array[i],Adds_ii)
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.I2M_array[0] = tmp[0]-Psum
        self.I2I_array[0] = tmp[1]-Psum
        for i in range(1,self.Match_num):
            tmp = np.zeros(2,dtype=np.float64)
            tmp[0] = np.logaddexp(self.I2M_array[i],Adds_im)
            tmp[1] = np.logaddexp(self.I2I_array[i],Adds_ii)
            Psum = np.logaddexp.reduce(tmp,axis=0)
            self.I2M_array[i] = tmp[0]-Psum
            self.I2I_array[i] = tmp[1]-Psum

        tmp = np.zeros(2,dtype=np.float64)
        tmp[0] = np.logaddexp(self.I2E,Adds_IE)
        tmp[1] = np.logaddexp(self.I2I_array[-1],Adds_iitail)
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.I2E = tmp[0]- Psum
        self.I2I_array[-1] = tmp[1]- Psum
        self.D2M_array = np.full_like(self.D2M_array,np.log(1/2))
        self.D2D_array = np.full_like(self.D2D_array,np.log(1/2))

        Adds_Pi_M = self.trProbAdds_PiM
        Adds_Pi_I = self.trProbAdds_PiI
        Adds_Pi_D = self.trProbAdds_PiD

        gamma_start_M_list = []
        gamma_start_I_list = []
        for i in self.graphStartNodes:
            gamma_start_M_list.append(self.gamma_M[i])
            gamma_start_I_list.append(self.gamma_I[i])
        gamma_start_M = np.logaddexp.reduce(gamma_start_M_list,axis=0)
        gamma_start_I = np.logaddexp.reduce(gamma_start_I_list,axis=0)
        self.pi_M = gamma_start_M[0]-np.logaddexp.reduce([gamma_start_M[0],gamma_start_I[0],self.gamma_D[-1][0]],axis=0)
        self.pi_I = gamma_start_I[0]-np.logaddexp.reduce([gamma_start_M[0],gamma_start_I[0],self.gamma_D[-1][0]],axis=0)
        self.pi_D = self.gamma_D[-1][0]-np.logaddexp.reduce([gamma_start_M[0],gamma_start_I[0],self.gamma_D[-1][0]],axis=0)
        tmp = np.zeros(3,dtype=np.float64)
        tmp[0] = np.logaddexp(self.pi_M,Adds_Pi_M)
        tmp[1] = np.logaddexp(self.pi_D,Adds_Pi_D)
        tmp[2] = np.logaddexp(self.pi_I,Adds_Pi_I)
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.pi_M = tmp[0] -Psum
        self.pi_D = tmp[1] -Psum
        self.pi_I = tmp[2] -Psum


        self.train_times+=1

        self.Me_Matrix_degenerate_base=[]
        self.Ie_Matrix_degenerate_base=[]
        for i in ['A','T','C','G']:
            self.Me_Matrix_degenerate_base.append(self.Me_Matrix[:,self.commonBaseDict[i]])
            self.Ie_Matrix_degenerate_base.append(self.Ie_Matrix[:,self.commonBaseDict[i]])
        for base in ["R","Y","M","K","S","W","H","B","V","D","N"]:
            degenerate_base = self.degenerateBaseDictionary[base]
            self.Ie_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Ie_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))
            self.Me_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Me_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))

        self.Me_Matrix_degenerate_base = np.array(self.Me_Matrix_degenerate_base)
        self.Ie_Matrix_degenerate_base = np.array(self.Ie_Matrix_degenerate_base)
        np.nan_to_num(self.Me_Matrix_degenerate_base)
        np.nan_to_num(self.Ie_Matrix_degenerate_base)

        print('ini save in ',self.train_times)
        endtime = datetime.now()
        print('use time',endtime-starttime)
        

        parameterDict={}
        parameterDict['_mm'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_mm'][1:-1] = self.M2M_array
        parameterDict['_mm'][0] = self.pi_M
        parameterDict['_mm'][-1] = self.M2E
        parameterDict['_dm'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_dm'][1:-1] = self.D2M_array
        parameterDict['_dm'][-1] = self.D2E
        parameterDict['_im'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_im'][:-1] = self.I2M_array
        parameterDict['_im'][-1]=self.I2E
        parameterDict['_mi'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_mi'][1:] = self.M2I_array
        parameterDict['_mi'][0] = self.pi_I
        parameterDict['_md'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_md'][1:-1] = self.M2D_array
        parameterDict['_md'][0] = self.pi_D
        
        parameterDict['_dd'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_dd'][1:-1] = self.D2D_array
        parameterDict['_di'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_di'][1:] = self.D2I_array
        parameterDict['_ii'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_ii'] = self.I2I_array
        parameterDict['_id'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_id'][:-1]=self.I2D_array
        parameterDict['match_emission'] = np.full((self.Match_num,4),np.NINF)
        parameterDict['match_emission'] = self.Me_Matrix
        parameterDict['insert_emission'] = np.full((self.Match_num+1,4),np.NINF)
        parameterDict['insert_emission'] = self.Ie_Matrix
        np.save(self.train_DAG_Path+'ini/'+str(self.parameterName)+'_pc_'+str(self.train_times)+'.npy',parameterDict)
        
    def mstep_with_random(self,starttime):
        # Insert 发射概率矩阵更新
        self.Ie_Matrix = self.gamma_o_I-self.IEi.reshape(-1, 1)
        # Match 发射概率矩阵更新
        self.Me_Matrix = self.gamma_o_M-self.MEi.reshape(-1, 1)

        # Match发射概率矩阵增加值（用于平滑）
        adds = np.array([self.emProbAdds_Match]*(self.normalBaseCount),dtype=np.float64)
        head_adds = np.array([self.emProbAdds_Match_head]*(self.normalBaseCount),dtype=np.float64)
        tail_adds = np.array([self.emProbAdds_Match_tail]*(self.normalBaseCount),dtype=np.float64)
        head_length = self.head_length
        tail_length = self.tail_length

        for i in range(self.Match_num):
            if i<=head_length:
                Madds = head_adds
            elif i>=self.Match_num-tail_length:
                Madds = tail_adds
            else:
                Madds = adds

            # # 1 此处替换为shape为self.normalBaseCount的随机array即可，记得log一下
            # print('***啊啊啊111')
            # random_array = np.log(0)    # 注意如果你想随机数置零，此处也要置零。或者你将random_array和下面中间tmp加入随机行注释掉
            random_array = np.log(np.random.rand(self.normalBaseCount))

            # 平滑
            tmp = np.logaddexp(self.Me_Matrix[i],Madds)
            # # 加入随机
            tmp = np.logaddexp(tmp,random_array)
            # 归一化
            Psum = np.logaddexp.reduce(tmp,axis=0)
            self.Me_Matrix[i] = tmp-Psum

        # 以上Match状态更新已完成

        # INsert概率目前固定为1/4，你可以尝试修改（模仿Match概率矩阵，或者自己有想法都可以）
        self.Ie_Matrix = np.full_like(self.Ie_Matrix,np.log(1/4))

        # 从Match状态出发的三个转移概率更新
        self.M2M_array = self.E_MM-self.MEi[:-1]
        self.M2I_array = self.E_MI-self.MEi
        self.M2D_array = self.E_MD-self.MEi[:-1]
        # 从Match状态出发的三个转移概率的平滑常数
        Adds_m_ = [self.trProbAdds_mm,self.trProbAdds_mi,self.trProbAdds_md]

        for i in range(self.Match_num-1):
            Adds = Adds_m_
            # 平滑
            tmp = np.zeros(3,dtype=np.float64)
            tmp[0] = np.logaddexp(self.M2M_array[i],Adds[0])
            tmp[1] = np.logaddexp(self.M2I_array[i],Adds[1])
            tmp[2] = np.logaddexp(self.M2D_array[i],Adds[2])
            # # 2 分别为三个转移概率加入随机值 # # '替换，记得log一下'
            # print('***啊啊啊222')
            random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_3 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            tmp[0] = np.logaddexp(tmp[0],random_1)
            tmp[1] = np.logaddexp(tmp[1],random_2)
            tmp[2] = np.logaddexp(tmp[2],random_3)

            # 归一化
            Psum = np.logaddexp.reduce(tmp,axis=0)
            self.M2M_array[i] = tmp[0] - Psum
            self.M2I_array[i] = tmp[1] - Psum
            self.M2D_array[i] = tmp[2] - Psum

        # Match到end和最后一个Match到Insert状态的概率平滑
        tmp = np.zeros(2,dtype=np.float64)
        tmp[0] = np.logaddexp(self.M2E,self.trProbAdds_mend)
        tmp[1] = np.logaddexp(self.M2I_array[-1],self.trProbAdds_mi_tail)

        # # 3 加入随机数 # # '替换，记得log一下'
        # print('***啊啊啊333')
        random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        tmp[0] = np.logaddexp(tmp[0],random_1)
        tmp[1] = np.logaddexp(tmp[1],random_2)

        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.M2E = tmp[0]- Psum
        self.M2I_array[-1] = tmp[1]- Psum

        # Insert状态出发的转移概率更新
        self.I2D_array = self.E_ID-self.IEi[:-1]
        self.I2I_array[1:-1] = self.E_II[1:-1]-self.IEi[1:-1]
        self.I2M_array[1:] = self.E_IM[1:]-self.IEi[1:-1]
        # Insert状态出发的转移概率平滑常数
        Adds_im =  self.trProbAdds_im
        Adds_ii =  self.trProbAdds_ii
        Adds_IE =  self.trProbAdds_iend
        Adds_iitail = self.trProbAdds_ii_tail

        for i in range(self.Match_num):
            tmp = np.zeros(2,dtype=np.float64)
            tmp[0] = np.logaddexp(self.I2M_array[i],Adds_im)
            tmp[1] = np.logaddexp(self.I2I_array[i],Adds_ii)

            # # 4 加入随机数 # # '替换，记得log一下'
            # print('***啊啊啊444')
            random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            tmp[0] = np.logaddexp(tmp[0],random_1)
            tmp[1] = np.logaddexp(tmp[1],random_2)

            #归一化
            Psum = np.logaddexp.reduce(tmp,axis=0)
            self.I2M_array[i] = tmp[0]-Psum
            self.I2I_array[i] = tmp[1]-Psum

        # INsert到End和最后一个Insert到Insert的概率
        tmp = np.zeros(2,dtype=np.float64)

        tmp[0] = np.logaddexp(self.I2E,Adds_IE)
        tmp[1] = np.logaddexp(self.I2I_array[-1],Adds_iitail)

        # # 5 加入随机数 # # '替换，记得log一下'
        # print('***啊啊啊555')
        random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        tmp[0] = np.logaddexp(tmp[0],random_1)
        tmp[1] = np.logaddexp(tmp[1],random_2)
        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.I2E = tmp[0]- Psum
        self.I2I_array[-1] = tmp[1]- Psum

        # Delate状态目前固定为1/2，可以改
        self.D2M_array = np.full_like(self.D2M_array,np.log(1/2))
        self.D2D_array = np.full_like(self.D2D_array,np.log(1/2))

        # Start状态出发的转移概率平滑常数
        Adds_Pi_M = self.trProbAdds_PiM
        Adds_Pi_I = self.trProbAdds_PiI
        Adds_Pi_D = self.trProbAdds_PiD
        # Start状态出发的转移概率更新
        gamma_start_M_list = []
        gamma_start_I_list = []
        for i in self.graphStartNodes:
            gamma_start_M_list.append(self.gamma_M[i])
            gamma_start_I_list.append(self.gamma_I[i])
        gamma_start_M = np.logaddexp.reduce(gamma_start_M_list,axis=0)
        gamma_start_I = np.logaddexp.reduce(gamma_start_I_list,axis=0)
        self.pi_M = gamma_start_M[0]-np.logaddexp.reduce([gamma_start_M[0],gamma_start_I[0],self.gamma_D[-1][0]],axis=0)
        self.pi_I = gamma_start_I[0]-np.logaddexp.reduce([gamma_start_M[0],gamma_start_I[0],self.gamma_D[-1][0]],axis=0)
        self.pi_D = self.gamma_D[-1][0]-np.logaddexp.reduce([gamma_start_M[0],gamma_start_I[0],self.gamma_D[-1][0]],axis=0)
        
        # 平滑
        tmp = np.zeros(3,dtype=np.float64)
        tmp[0] = np.logaddexp(self.pi_M,Adds_Pi_M)
        tmp[1] = np.logaddexp(self.pi_D,Adds_Pi_D)
        tmp[2] = np.logaddexp(self.pi_I,Adds_Pi_I)
        # 分别为三个转移概率加入随机值 # # 6 '替换，记得log一下'
        # print('***啊啊啊666')
        random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        random_3 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
        tmp[0] = np.logaddexp(tmp[0],random_1)
        tmp[1] = np.logaddexp(tmp[1],random_2)
        tmp[2] = np.logaddexp(tmp[2],random_3)
        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        self.pi_M = tmp[0] -Psum
        self.pi_D = tmp[1] -Psum
        self.pi_I = tmp[2] -Psum


        self.train_times+=1

        self.Me_Matrix_degenerate_base=[]
        self.Ie_Matrix_degenerate_base=[]
        for i in ['A','T','C','G']:
            self.Me_Matrix_degenerate_base.append(self.Me_Matrix[:,self.commonBaseDict[i]])
            self.Ie_Matrix_degenerate_base.append(self.Ie_Matrix[:,self.commonBaseDict[i]])
        for base in ["R","Y","M","K","S","W","H","B","V","D","N"]:
            degenerate_base = self.degenerateBaseDictionary[base]
            self.Ie_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Ie_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))
            self.Me_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Me_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))

        self.Me_Matrix_degenerate_base = np.array(self.Me_Matrix_degenerate_base)
        self.Ie_Matrix_degenerate_base = np.array(self.Ie_Matrix_degenerate_base)
        np.nan_to_num(self.Me_Matrix_degenerate_base)
        np.nan_to_num(self.Ie_Matrix_degenerate_base)

        print('ini save in ',self.train_times)
        endtime = datetime.now()
        print('use time',endtime-starttime)
        

        parameterDict={}
        parameterDict['_mm'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_mm'][1:-1] = self.M2M_array
        parameterDict['_mm'][0] = self.pi_M
        parameterDict['_mm'][-1] = self.M2E
        parameterDict['_dm'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_dm'][1:-1] = self.D2M_array
        parameterDict['_dm'][-1] = self.D2E
        parameterDict['_im'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_im'][:-1] = self.I2M_array
        parameterDict['_im'][-1]=self.I2E
        parameterDict['_mi'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_mi'][1:] = self.M2I_array
        parameterDict['_mi'][0] = self.pi_I
        parameterDict['_md'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_md'][1:-1] = self.M2D_array
        parameterDict['_md'][0] = self.pi_D
        
        parameterDict['_dd'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_dd'][1:-1] = self.D2D_array
        parameterDict['_di'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_di'][1:] = self.D2I_array
        parameterDict['_ii'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_ii'] = self.I2I_array
        parameterDict['_id'] = np.full(self.Match_num+1,np.NINF)
        parameterDict['_id'][:-1]=self.I2D_array
        parameterDict['match_emission'] = np.full((self.Match_num,4),np.NINF)
        parameterDict['match_emission'] = self.Me_Matrix
        parameterDict['insert_emission'] = np.full((self.Match_num+1,4),np.NINF)
        parameterDict['insert_emission'] = self.Ie_Matrix
        np.save(self.train_DAG_Path+'ini/'+str(self.parameterName)+'_pc_'+str(self.train_times)+'.npy',parameterDict)
    
    
    def init_train_data(self,Viterbi_DAG_Path,ref_node_list,ref_seq,modify_dict,kill_degenerate_base =True,windows_length=100):
        self.windows_length = windows_length
        self.head_length = modify_dict['head_length']
        self.tail_length = modify_dict['tail_length']
        self.emProbAdds_Match = modify_dict['emProbAdds_Match']
        self.emProbAdds_Match_head = modify_dict['emProbAdds_Match_head']
        self.emProbAdds_Match_tail = modify_dict['emProbAdds_Match_tail']
        self.trProbAdds_mm = modify_dict['trProbAdds_mm']
        self.trProbAdds_md = modify_dict['trProbAdds_md']
        self.trProbAdds_mi = modify_dict['trProbAdds_mi']
        self.trProbAdds_PiM = modify_dict['trProbAdds_PiM']
        self.trProbAdds_PiI = modify_dict['trProbAdds_PiI']
        self.trProbAdds_PiD = modify_dict['trProbAdds_PiD']
        self.trProbAdds_im = modify_dict['trProbAdds_im']
        self.trProbAdds_ii = modify_dict['trProbAdds_ii']
        self.trProbAdds_iend = modify_dict['trProbAdds_iend']
        self.trProbAdds_ii_tail = modify_dict['trProbAdds_ii_tail']
        self.trProbAdds_mend = modify_dict['trProbAdds_mend']
        self.trProbAdds_mi_tail = modify_dict['trProbAdds_mi_tail']
        
        Trace_path = np.load(Viterbi_DAG_Path+'Traceability_path.npy',allow_pickle=True)
        self.DAG = load_graph(Viterbi_DAG_Path)
        for node in self.DAG.nodeList:
            node[3]=set([n[1] for n in Trace_path[node[0]]])

        new_ref_list=['x']*(len(ref_node_list)-1)
        ref_node_set=set(ref_node_list[1:])
        for node in self.DAG.nodeList:
            co_set=node[3]&ref_node_set
            if co_set:
                idx = ref_node_list[1:].index(co_set.pop())
                new_ref_list[idx]=node[0]
        ref_node_list = ['x']*(len(ref_seq)-len(new_ref_list))+new_ref_list
        self.DAG.fragmentReduce(1)

        self.DAG.calculateStateRange(ref_node_list)

        print('endNodeSet',len(self.DAG.endNodeSet))
        print('startNodeSet',len(self.DAG.startNodeSet))
        if kill_degenerate_base:
            print('kill')
            Flag = self.DAG.removeDegenerateBasePaths()

            if Flag:
                idmapping= {value: key for key, value in self.DAG.id_mapping.items()}
            else:
                idmapping= {i: i for i in range(self.DAG.totalNodes)}
        else:
            print('not kill')
            idmapping= {i: i for i in range(self.DAG.totalNodes)}

        
        
        print(self.Match_num)
        print(len(ref_node_list))
        print('endNodeSet',len(self.DAG.endNodeSet))
        print('startNodeSet',len(self.DAG.startNodeSet))
        print(self.windows_length)


        self.maxrange=0
        self.stateRangeDict = []
        self.range_length=0
        self.arrayRangeDict=[]
        for node in range(self.DAG.totalNodes):
            node = idmapping[node]
            
            self.stateRangeDict.append(max(0,self.DAG.ref_coor[node][0]-self.windows_length))
            self.stateRangeDict.append(min(len(ref_node_list),self.DAG.ref_coor[node][1]+self.windows_length))

            sted=[self.range_length]
            self.range_length+=(self.stateRangeDict[-1]-self.stateRangeDict[-2]+1)
            sted.append(self.range_length)
            self.arrayRangeDict.append(sted)
            if self.stateRangeDict[-1]-self.stateRangeDict[-2]>self.maxrange:
                self.maxrange=self.stateRangeDict[-1]-self.stateRangeDict[-2]
        print(self.range_length,'RANGE LENGTH')
        sted=[self.range_length]
        self.range_length+=(self.maxrange+1)
        sted.append(self.range_length)
        self.arrayRangeDict.append(sted)

        self.vnum=self.DAG.sequenceNum
        # if 'zip20/' in Viterbi_DAG_Path:
        #     vtuple = np.load(Viterbi_DAG_Path.replace('zip20/','')+'v_id.npy').tolist()
        # elif 'zip9/' in Viterbi_DAG_Path:
        #     vtuple = np.load(Viterbi_DAG_Path.replace('zip9/','')+'v_id.npy').tolist()
        vtuple = np.load(Viterbi_DAG_Path+'v_id.npy').tolist()
        self.all_virus = {tu[1] for tu in vtuple}
        self.vlist = [tu[1] for tu in vtuple]
        self.v2id_dict = dict(vtuple)
        self.id2v_dict = {value: key for key, value in self.v2id_dict.items()}


        width=[]
        for i in range(1000,self.DAG.maxlength-1000,int((self.DAG.maxlength-2000)/5)):
            print(i)
            width.append(self.DAG.coordinateList.count(i))
        print(width)
        self.pool_num = int(np.mean(width)//2)
        self.pool_num = min(max(1,self.pool_num),35)
        print('pool_num:',self.pool_num)

        self.clique_list,self.clique_link = self.DAG.coarse_grained_graph_construction()
        self.graphStartNodes=set()
        self.graphEndNodes=set()
        # self.graph_start_clique=self.DAG.
        for node in self.DAG.nodeList:
            if self.DAG.findParentNodes(node[0])==[]:
                self.graphStartNodes.add(node[0])
            if self.DAG.findChildNodes(node[0])==[]:
                self.graphEndNodes.add(node[0])
        self.sequenceNum = self.DAG.sequenceNum


        

    def fit(self,Viterbi_DAG_Path,ref_node_list,ref_seq,modify_dict,kill_degenerate_base=True,windows_length=100):
        
        self.init_train_data(Viterbi_DAG_Path,ref_node_list,ref_seq,modify_dict,kill_degenerate_base,windows_length)

        print(self.sequenceNum,self.DAG.totalNodes,len(self.graphStartNodes),len(self.graphEndNodes))
        oriprob = np.NINF
        # if mode == 'memory':
        shared_array_alphaM = Array('d', self.range_length)
        alpha_Matrix_M = np.frombuffer(shared_array_alphaM.get_obj(), dtype=np.float64)
        alpha_Matrix_M = alpha_Matrix_M.reshape(self.range_length)

        shared_array_alphaI = Array('d', self.range_length)
        alpha_Matrix_I = np.frombuffer(shared_array_alphaI.get_obj(), dtype=np.float64)
        alpha_Matrix_I = alpha_Matrix_I.reshape(self.range_length)

        shared_array_alphaD = Array('d', self.range_length)
        alpha_Matrix_D = np.frombuffer(shared_array_alphaD.get_obj(), dtype=np.float64)
        alpha_Matrix_D = alpha_Matrix_D.reshape(self.range_length)

        shared_array_betaM = Array('d', self.range_length)
        beta_Matrix_M = np.frombuffer(shared_array_betaM.get_obj(), dtype=np.float64)
        beta_Matrix_M = beta_Matrix_M.reshape(self.range_length)

        shared_array_betaI = Array('d', self.range_length)
        beta_Matrix_I = np.frombuffer(shared_array_betaI.get_obj(), dtype=np.float64)
        beta_Matrix_I = beta_Matrix_I.reshape(self.range_length)

        shared_array_betaD = Array('d', self.range_length)
        beta_Matrix_D = np.frombuffer(shared_array_betaD.get_obj(), dtype=np.float64)
        beta_Matrix_D = beta_Matrix_D.reshape(self.range_length)

        shared_array_leftbetaM = Array('d', self.range_length)
        left_beta_Matrix_M = np.frombuffer(shared_array_leftbetaM.get_obj(), dtype=np.float64)
        left_beta_Matrix_M = left_beta_Matrix_M.reshape(self.range_length)

        shared_array_leftbetaI = Array('d', self.range_length)
        left_beta_Matrix_I = np.frombuffer(shared_array_leftbetaI.get_obj(), dtype=np.float64)
        left_beta_Matrix_I = left_beta_Matrix_I.reshape(self.range_length)
        

        while True:
            alpha_Matrix_M[:] = np.NINF
            alpha_Matrix_D[:] = np.NINF
            alpha_Matrix_I[:] = np.NINF
            beta_Matrix_M[:] = np.NINF
            beta_Matrix_D[:] = np.NINF
            beta_Matrix_I[:] = np.NINF
            left_beta_Matrix_M[:] = np.NINF
            left_beta_Matrix_I[:] = np.NINF
            
            prob,starttime = self.estep(alpha_Matrix_M,alpha_Matrix_I,alpha_Matrix_D,beta_Matrix_M,beta_Matrix_D,beta_Matrix_I,left_beta_Matrix_M,left_beta_Matrix_I)
            print('Train {} prob advance'.format(self.train_times),prob-oriprob)
            if prob>-0.5 or prob-oriprob<0.1: 
                break

            self.mstep(starttime)
            # self.mstep_with_random(starttime)  # # 6.5 改成random版本的
            
            if prob-oriprob<0.1:
                break

            oriprob = prob
        print('save ini in {}'.format(self.train_times))

    def init_viterbi_data(self,Viterbi_DAG_Path,ref_node_list,ref_seq,onm2db=False,polyA=False,windows_length=100):
        self.windows_length = windows_length
        self.Viterbi_DAG_Path = Viterbi_DAG_Path
        Trace_path = np.load(Viterbi_DAG_Path+'Traceability_path.npy',allow_pickle=True)
        self.DAG = load_graph(Viterbi_DAG_Path,mode='ali')
        Trace_nodes=[]
        for node in self.DAG.nodeList:
            Trace_nodes.append(set([n[1] for n in Trace_path[node[0]]]))

        
        new_ref_list=['x']*(len(ref_node_list)-1)

        ref_node_set=set(ref_node_list[1:])
        for node in self.DAG.nodeList:
            co_set=Trace_nodes[node[0]]&ref_node_set
            if co_set:
                idx = ref_node_list[1:].index(co_set.pop())
                new_ref_list[idx]=node[0]
        ref_node_list = ['x']*(len(ref_seq)-len(new_ref_list))+new_ref_list

        self.DAG.fragmentReduce(1)
        
        self.DAG.calculateStateRange(ref_node_list)
        self.vnum=self.DAG.sequenceNum

        vtuple = np.load(Viterbi_DAG_Path+'v_id.npy').tolist()
        self.all_virus = {tu[1] for tu in vtuple}
        self.vlist = [tu[1] for tu in vtuple]
        self.v2id_dict = dict(vtuple)
        self.id2v_dict = {value: key for key, value in self.v2id_dict.items()}

        width=[]
        for i in range(1000,self.DAG.maxlength-1000,int((self.DAG.maxlength-2000)/5)):
            width.append(self.DAG.coordinateList.count(i))
        self.pool_num = int(np.mean(width)//3)
        self.pool_num = max(self.pool_num,1)
        if onm2db:
            ori_node_list = []
            for node in self.DAG.nodeList:
                ori_node_list.append(np.array(node[3],dtype=np.uint32))
            GRAPH.onm2db(Viterbi_DAG_Path,ori_node_list)
            pass
        print('pool_num:',self.pool_num)
        self.DAG.totalNodes = self.DAG.totalNodes
        


        self.graphStartNodes = self.DAG.startNodeSet
        self.graphEndNodes = self.DAG.endNodeSet
        self.sequenceNum = self.DAG.sequenceNum
        self.maxrange=0
        self.stateRangeDict = []
        self.range_length=0
        self.arrayRangeDict=[]
        for node in range(self.DAG.totalNodes):            
            self.stateRangeDict.append(max(0,self.DAG.ref_coor[node][0]-self.windows_length))
            self.stateRangeDict.append(min(len(ref_node_list),self.DAG.ref_coor[node][1]+self.windows_length))

            sted=[self.range_length]
            self.range_length+=(self.stateRangeDict[-1]-self.stateRangeDict[-2]+1)
            sted.append(self.range_length)

            self.arrayRangeDict.append(sted)
            if self.stateRangeDict[-1]-self.stateRangeDict[-2]>self.maxrange:
                self.maxrange=self.stateRangeDict[-1]-self.stateRangeDict[-2]
                
        print('Mean state range length: ',int(self.range_length/(len(self.stateRangeDict)/2)))
        print('Max state range length: ',self.maxrange)
        sted=[self.range_length]
        self.range_length+=(self.maxrange)
        sted.append(self.range_length)
        self.arrayRangeDict.append(sted)

        self.stateRangeDict = np.array(self.stateRangeDict,dtype=np.uint16)
        if polyA==True:
            self.M2I_array[-1]=np.log(0.1)
            self.M2E=np.log(0.9)
        self.clique_list,self.clique_link = self.DAG.coarse_grained_graph_construction()
        self.ref_seq = ref_seq

    
        
    def Viterbi(self,seqiddb,mode = 'memory'):
        
        def write_hiddensates_head(indexlist):
            decrease_list=[-1,-1,0]
            for index in indexlist:
                clique = self.clique_list[index]
                node = clique.pop()
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                delta_all = [delta_M_mem[arrayRangeEnd-2],delta_D_mem[arrayRangeEnd-2],delta_I_mem[arrayRangeEnd-1]]
                delta_index_start,delta_index_end = self.stateRangeDict[2*node:2*node+2]
                matrix_length=delta_index_end-delta_index_start
                nowstate = np.argmax([delta_all[0]+self.M2E,delta_all[1]+self.D2E,delta_all[2]+self.I2E])
                nowstateindex = self.Match_num+decrease_list[nowstate]
                laststate=laststate_mem[3*arrayRangeStart:3*arrayRangeEnd]

                nowstate, nowstateindex = Profile_HMM.update_state(laststate,matrix_length,nowstate, nowstateindex, delta_index_start, decrease_list)
                
                if nowstate==2:
                    hidden_states[node] = {(nowstate,nowstateindex):['',1]}
                    ali[nowstateindex] = max([1,ali[nowstateindex]])
                else:
                    hidden_states[node] = {(nowstate,nowstateindex):['',0]}
                write_hiddensates(clique)
                # q.put(node)
                doneList[node]=0
                checklist=self.DAG.findParentClique(index)
                for i in checklist:
                    with lock:
                        outdegreeDict[i]-=1
                        if outdegreeDict[i]==0:
                            q.put(i)
        def write_hiddensates(nodes):
            is_head=True
            for node in nodes[::-1]:
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                children_nodes = self.DAG.findChildNodes(node)
                decrease_list=np.array([-1,-1,0])
                delta_index_start,delta_index_end = self.stateRangeDict[2*node:2*node+2]
                range_length = delta_index_end-delta_index_start
                tmp_state_dict={}
                laststate_local = laststate_mem[3*arrayRangeStart:3*arrayRangeEnd]
                delta_M,delta_D,delta_I = delta_M_mem[arrayRangeStart:arrayRangeEnd-1],delta_D_mem[arrayRangeStart:arrayRangeEnd-1],delta_I_mem[arrayRangeStart:arrayRangeEnd]
                for fnode in children_nodes:

                    _arrayRangeStart,_arrayRangeEnd = self.arrayRangeDict[fnode]
                    delta_index_start_f,delta_index_end_f = self.stateRangeDict[2*fnode:2*fnode+2]
                    range_length_f = delta_index_end_f - delta_index_start_f
                    orinode = orinode_mem[3*_arrayRangeStart:3*_arrayRangeEnd]
                    laststate = laststate_mem[3*_arrayRangeStart:3*_arrayRangeEnd]
                    children_node_hiddenstate = hidden_states[fnode]
                    for oristate in children_node_hiddenstate.keys():
                        nowstate = oristate[0]
                        nowstateindex = oristate[1]

                        fm_index=nowstate*range_length_f+nowstateindex-delta_index_start_f
                        try:
                            ori_node = orinode[fm_index]
                        except:
                            ori_node=-1
                        from_state = [(fnode,oristate)]
                        ori_step = children_node_hiddenstate[oristate][1]
                        if ori_node==node:
                            nowstate = laststate[fm_index]
                            nowstateindex += decrease_list[nowstate]
                            nowstate, nowstateindex = Profile_HMM.update_state(laststate_local,range_length,nowstate, nowstateindex, delta_index_start, decrease_list)

                        else:

                            if nowstateindex-delta_index_start == 0:
                                
                                nowstate=2
                                
                            else:
                                indexs=nowstateindex-delta_index_start+decrease_list
                                
                                try:
                                    delta_all = np.array([delta_M[indexs[0]],delta_D[indexs[1]],delta_I[indexs[2]]])
                                except:
                                    delta_all = np.array([-1,-1,-1])
                                        # print(node,nowstate,nowstateindex,'b')
                                P_T =  np.fromiter((Adict[(laststate,nowstate)][nowstateindex+decrease_list[laststate]] for laststate in range(3)),dtype=np.float64)
                                nowstate = np.argmax(delta_all+P_T)

                            nowstateindex+=decrease_list[nowstate]
                            nowstate, nowstateindex = Profile_HMM.update_state(laststate_local,range_length,nowstate, nowstateindex, delta_index_start, decrease_list)
                        state = (nowstate,nowstateindex)

                        tmp_state_dict[state] = tmp_state_dict.get(state,[[],[]])
                        tmp_state_dict[state][0].extend(from_state)
                        tmp_state_dict[state][1].append(int((ori_step+1)*(nowstate/2)))

                if len(tmp_state_dict.keys())==1:
                    new_tmp_state_dict={}
                    state = next(iter(tmp_state_dict))
                    if state[1]<0:
                        print('error',node,state,self.DAG.findChildNodes(node),[hidden_states[node] for node in self.DAG.findChildNodes(node)])

                    if state[0]==2:
                        new_tmp_state_dict[state] = ['', max(tmp_state_dict[state][1])]
                        lock.acquire()
                        ali[state[1]] = max([new_tmp_state_dict[state][1],ali[state[1]]])
                        lock.release()
                    else:
                        new_tmp_state_dict[state] = ['', 0]
                else:
                    if is_head:
                        tmp_hiddenstates_dict={}
                        for childernnode in children_nodes:
                            tmp_hiddenstates_dict[childernnode] = hidden_states[childernnode]
                        content_virus = query_contend_sequence_id(node)
                        new_tmp_state_dict = {}
                        for state in tmp_state_dict.keys():
                            viruslist = []
                            allsize=0
                            for f in tmp_state_dict[state][0]:
                                fn = f[0]
                                fstate=f[1]
                                if isinstance(tmp_hiddenstates_dict[fn][fstate][0],str):
                                    virus = query_contend_sequence_id(fn)
                                    if virus.size!=0:
                                        viruslist.append(virus)
                                        allsize+=virus.shape[1]
                                else:
                                    viruslist.append(tmp_hiddenstates_dict[fn][fstate][0])
                                    allsize+=tmp_hiddenstates_dict[fn][fstate][0].shape[1]
                            virus_matrix = np.full((2,allsize),0,dtype=np.uint32)
                            if viruslist:
                                nct = Profile_HMM.get_intersection(content_virus,viruslist,virus_matrix,self.DAG.originalfragmentLength)
                                if nct.size!=0:
                                    if state[0]==2:
                                        new_tmp_state_dict[state] = [nct.copy(order='C'),max(tmp_state_dict[state][1])]
                                        lock.acquire()
                                        ali[state[1]] = max([new_tmp_state_dict[state][1],ali[state[1]]])
                                        lock.release()
                                    else:
                                        new_tmp_state_dict[state] = [nct.copy(order='C'),0]
                                
                    else:
                        tmp_hiddenstates_dict={}
                        for childernnode in children_nodes:
                            tmp_hiddenstates_dict[childernnode] = hidden_states[childernnode]
                        new_tmp_state_dict = {}
                        for state in tmp_state_dict.keys():
                            viruslist = []
                            allsize=0
                            for f in tmp_state_dict[state][0]:
                                fn = f[0]
                                fstate=f[1]
                                viruslist.append(tmp_hiddenstates_dict[fn][fstate][0])
                                allsize+=tmp_hiddenstates_dict[fn][fstate][0].shape[1]
                            virus_matrix = np.full((2,allsize),0,dtype=np.uint32)
                            nct = Profile_HMM.get_intersection_no_fork(viruslist,virus_matrix,self.DAG.originalfragmentLength)
                            if state[0]==2:
                                new_tmp_state_dict[state] = [nct,max(tmp_state_dict[state][1])]
                                lock.acquire()
                                ali[state[1]] = max([new_tmp_state_dict[state][1],ali[state[1]]])
                                lock.release()
                            else:
                                new_tmp_state_dict[state] = [nct,0]
                    if len(new_tmp_state_dict) == 1:
                        state = next(iter(new_tmp_state_dict))
                        new_tmp_state_dict[state] = ('', new_tmp_state_dict[state][1])

                hidden_states[node] = new_tmp_state_dict
                doneList[node]=0
                is_head = False

        
        
        def init_delta_head():
            delta_head_dict={}
            for baseID in range(self.Me_Matrix_degenerate_base.shape[0]):
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]                

                delta_M = np.full(self.Match_num, np.NINF, dtype=np.float64)
                delta_I = np.full(self.Match_num + 1, np.NINF, dtype=np.float64)
                delta_D = np.full(self.Match_num, np.NINF, dtype=np.float64)
                delta_M[0]=self.pi_M+Me[0]
                delta_I[0]=self.pi_I+Ie[0]
                last_delta_D = np.full(self.Match_num, np.NINF, dtype=np.float64)
                last_delta_D[:self.maxrange] = First_deata_D
                delta_M[1:] = last_delta_D[:-1]+self.D2M_array+Me[1:]
                delta_I[1:] = last_delta_D+self.D2I_array+Ie[1:]
                
                maxProbOrigin_D = np.array([0]*(self.Match_num),dtype='int')
                maxProbOrigin_D[0]=2
                delta_D[0]=delta_I[0]+self.I2D_array[0]
                for i in range(1,self.Match_num):
                    D_arg = [delta_M[i-1]+self.M2D_array[i-1],delta_D[i-1]+self.D2D_array[i-1],delta_I[i]+self.I2D_array[i]]
                    delta_D[i] = np.max(D_arg)
                    maxProbOrigin_D[i] = np.argmax(D_arg)
                maxProbOrigin_M = np.full(self.Match_num, 2, dtype='int')
                maxProbOrigin_M[0]=3
                maxProbOrigin_I = np.full(self.Match_num+1, 2, dtype='int')
                maxProbOrigin_I[0]=3
                delta_head_dict[baseID]=[delta_M,delta_I,delta_D,maxProbOrigin_M,maxProbOrigin_I,maxProbOrigin_D]
            return delta_head_dict
        def write_delta_head(indexlist,delta_head_dict):
            for index in indexlist:
                clique = self.clique_list[index]
                node = clique[0]
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict.get(self.DAG.nodeList[node][1][-1],14)
                values = delta_head_dict[baseID]
                stateRangeStart=self.stateRangeDict[2*node]
                stateRangeEnd=self.stateRangeDict[2*node+1]


                maxProbOrigin_M = values[3][stateRangeStart:stateRangeEnd]
                maxProbOrigin_I = values[4][stateRangeStart:stateRangeEnd+1]
                maxProbOrigin_D = values[5][stateRangeStart:stateRangeEnd]
                delta_M_mem[arrayRangeStart:arrayRangeEnd-1] = values[0][stateRangeStart:stateRangeEnd]
                delta_D_mem[arrayRangeStart:arrayRangeEnd-1] = values[2][stateRangeStart:stateRangeEnd]
                delta_I_mem[arrayRangeStart:arrayRangeEnd] = values[1][stateRangeStart:stateRangeEnd+1]

                laststate_mem[3*arrayRangeStart:3*arrayRangeEnd-2]= np.concatenate((maxProbOrigin_M,maxProbOrigin_D,maxProbOrigin_I))
                write_delta(clique[1:])
                doneList[node]=0
                checklist=self.DAG.findChildrenClique(index)
                for i in checklist:
                    with lock:
                        indegree_dict[i]-=1
                        if indegree_dict[i]==0:
                            q.put(i)
        def write_delta(nodes):
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            for node in nodes:
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict.get(self.DAG.nodeList[node][1][-1],14)
                partennodes = self.DAG.findParentNodes(node)
                parentNodeWeightList=[]
                alist = [self.DAG.edgeWeightDict[(lnode,node)] for lnode in partennodes]
                b = np.sum(alist)
                for lnode in partennodes:
                    a = self.DAG.edgeWeightDict[(lnode,node)]
                    adds=0
                    ab = (a+adds/len(partennodes))/(b+adds)
                    parentNodeWeightList.append([ab,lnode])
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                last_delta_M_list = np.full((len(parentNodeWeightList), self.Match_num), np.NINF, dtype=np.float64)
                last_delta_I_list = np.full((len(parentNodeWeightList), self.Match_num+1), np.NINF, dtype=np.float64)
                last_delta_D_list = np.full((len(parentNodeWeightList), self.Match_num), np.NINF, dtype=np.float64)

                for i, fnode in enumerate(parentNodeWeightList):
                    _arrayRangeStart,_arrayRangeEnd = self.arrayRangeDict[fnode[1]]
                    weight = np.log(fnode[0])
                    fathernode_left_limit = self.stateRangeDict[2*fnode[1]]
                    fathernode_right_limit = self.stateRangeDict[2*fnode[1]+1]
                    last_delta_M_list[i][fathernode_left_limit:fathernode_right_limit] = delta_M_mem[_arrayRangeStart:_arrayRangeEnd-1] + weight
                    last_delta_D_list[i][fathernode_left_limit:fathernode_right_limit] = delta_D_mem[_arrayRangeStart:_arrayRangeEnd-1] + weight
                    last_delta_I_list[i][fathernode_left_limit:fathernode_right_limit+1] = delta_I_mem[_arrayRangeStart:_arrayRangeEnd] + weight

                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                arrayLength =stateRangeEnd-stateRangeStart

                last_delta_M_list = last_delta_M_list[:,stateRangeStart:stateRangeEnd]
                last_delta_I_list = last_delta_I_list[:,stateRangeStart:stateRangeEnd+1]
                last_delta_D_list = last_delta_D_list[:,stateRangeStart:stateRangeEnd]
                last_delta_M = np.max(last_delta_M_list,axis=0) 
                last_delta_I = np.max(last_delta_I_list,axis=0) 
                last_delta_D = np.max(last_delta_D_list,axis=0)
                lm=[parentNodeWeightList[idx][1] for idx in  np.argmax(last_delta_M_list,axis=0)]
                li=[parentNodeWeightList[idx][1] for idx in  np.argmax(last_delta_I_list,axis=0)]
                ld=[parentNodeWeightList[idx][1] for idx in  np.argmax(last_delta_D_list,axis=0)]


                
                delta_I, delta_M, delta_D, maxProbOrigin_I, maxProbOrigin_M, maxProbOrigin_D = Profile_HMM.calculate_delta_values(stateRangeStart, stateRangeEnd, arrayLength,
                                                                            last_delta_I, last_delta_M, last_delta_D,
                                                                            I2I_array, I2M_array, M2I_array, M2M_array, D2I_array, D2M_array, I2D_array, D2D_array,
                                                                            Ie, Me,M2D_array)

                delta_M_mem[arrayRangeStart:arrayRangeEnd-1] = delta_M

                delta_D_mem[arrayRangeStart:arrayRangeEnd-1] = delta_D
                delta_I_mem[arrayRangeStart:arrayRangeEnd] = delta_I

                st=3*arrayRangeStart
                laststate_mem[st:st+arrayLength]=maxProbOrigin_M
                st+=arrayLength
                laststate_mem[st:st+arrayLength]=maxProbOrigin_D
                st+=arrayLength
                laststate_mem[st:st+arrayLength+1]=maxProbOrigin_I

                st=3*arrayRangeStart
                orinode_mem[st:st+arrayLength]=lm
                st+=arrayLength
                orinode_mem[st:st+arrayLength]=ld
                st+=arrayLength
                orinode_mem[st:st+arrayLength+1]=li
                doneList[node]=0

        def calculate_delta(goon_flag,lock,nodelist,delta_head_dict):
            clique_list = self.clique_list
            write_delta_head(nodelist,delta_head_dict)
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1) # timeout=8
                except:
                    
                    continue
                while start_node!=None:
                    clique = clique_list[start_node]
                    write_delta(clique)
                    checklist=self.DAG.findChildrenClique(start_node)
                    todolist=[]
                    for i in checklist:
                        with lock:
                            indegree_dict[i]-=1
                            if indegree_dict[i]==0:
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()
                        for i in todolist:
                            q.put(i)
                    else:
                        start_node=None


        def calculate_state(goon_flag,lock,nodelist):
            clique_list = self.clique_list
            write_hiddensates_head(nodelist)
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1)
                except:
                    
                    continue
                while start_node!=None:
                    clique = clique_list[start_node]
                    write_hiddensates(clique)
                    checklist=self.DAG.findParentClique(start_node)
                    todolist=[]
                    for i in checklist:
                        with lock:
                            outdegreeDict[i]-=1
                            if outdegreeDict[i]==0:
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()
                        for i in todolist:
                            q.put(i)
                    else:
                        start_node=None

        def is_end_f(goon_flag):
            start = time.time()
            while True:
                time.sleep(1)
                runtime = time.time() - start
                percent = np.round((self.DAG.totalNodes-np.count_nonzero(doneList))/self.DAG.totalNodes,5)
                bar = ('#' * int(percent * 20)).ljust(20)
                mins, secs = divmod(runtime, 60)
                time_format = '{:02d}:{:02d}'.format(int(mins), int(secs))
                sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}% 完成 (运行时间: {time_format} 分钟)')
                sys.stdout.flush()
                if 0==np.count_nonzero(doneList>0):
                    goon_flag.value=0
                    
                    break

        def is_end_b(goon_flag):
            start = time.time()
            while True:
                time.sleep(1)
                runtime = time.time() - start
                percent = np.round((self.DAG.totalNodes-np.count_nonzero(doneList))/self.DAG.totalNodes,5)
                bar = ('#' * int(percent * 20)).ljust(20)
                mins, secs = divmod(runtime, 60)
                time_format = '{:02d}:{:02d}'.format(int(mins), int(secs))
                sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}% 完成 (运行时间: {time_format} 分钟)')
                sys.stdout.flush()

                if 0==np.count_nonzero(doneList>0):
                    goon_flag.value=0
                    
                    break

        def query_contend_sequence_id(node):
            return seqiddb.findSequenceSource(self.DAG.nodeList[node][3],self.DAG.firstBiteofONM,self.DAG.allBiteofONM)

        
        if mode =='memory':
            tmp_delta_M_mem = np.full(self.range_length, np.NINF)
            delta_M_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*np.dtype(np.float64).itemsize)
            delta_M_mem =np.ndarray(self.range_length, dtype=np.float64, buffer=delta_M_mem_shm.buf)
            delta_M_mem[:] = tmp_delta_M_mem
            tmp_delta_I_mem = np.full(self.range_length, np.NINF)
            delta_I_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*np.dtype(np.float64).itemsize)
            delta_I_mem =np.ndarray(self.range_length, dtype=np.float64, buffer=delta_I_mem_shm.buf)
            delta_I_mem[:] = tmp_delta_I_mem
            tmp_delta_D_mem = np.full(self.range_length, np.NINF)
            delta_D_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*np.dtype(np.float64).itemsize)
            delta_D_mem =np.ndarray(self.range_length, dtype=np.float64, buffer=delta_D_mem_shm.buf)
            delta_D_mem[:] = tmp_delta_D_mem
            laststate_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*3*np.dtype(np.uint8).itemsize)
            laststate_mem = np.ndarray(self.range_length*3, dtype=np.uint8, buffer=laststate_mem_shm.buf)
            laststate_mem[:] = 2
            orinode_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*3*np.dtype('int').itemsize)
            orinode_mem = np.ndarray(self.range_length*3, dtype=int, buffer=orinode_mem_shm.buf)

            
        

        Adict = {(0,0):self.M2M_array,(0,2):self.M2I_array,(0,1):self.M2D_array,(2,0):self.I2M_array,(1,2):self.D2I_array,(2,1):self.I2D_array,(2,2):self.I2I_array,(1,0):self.D2M_array,(1,1):self.D2D_array}
        arrayRangeStart=self.arrayRangeDict[-1][0]
        delta_D_mem[arrayRangeStart] = self.pi_D
        for i in range(1,self.maxrange):
            delta_D_mem[arrayRangeStart+i] = delta_D_mem[arrayRangeStart+i-1]+self.D2D_array[i-1]
        First_deata_D = delta_D_mem[arrayRangeStart:]

        goon_flag = Value('i',1)
        # working_bots = Value('i',self.pool_num)

        indegree_dict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.clique_num*np.dtype(np.uint16).itemsize)
        indegree_dict =np.ndarray((self.DAG.clique_num,), dtype=np.int16, buffer=indegree_dict_shm.buf)
        indegree_dict[:] = np.zeros(self.DAG.clique_num)
        for link in self.clique_link:
            indegree_dict[link[1]]+=1
        doneList_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint8).itemsize)
        doneList =np.ndarray((self.DAG.totalNodes,), dtype=np.uint8, buffer=doneList_shm.buf)
        doneList[:] = np.full(self.DAG.totalNodes,1)

        lock = Lock()
        # pool_lock = Lock()

        v_dict={}
        all_v = sorted(self.all_virus, key=lambda x: (int(x.split("_")[0]), int(x.split("_")[-1])))
        namelist=[]
        namedict={}
        v_num = len(all_v)
        for i in range(v_num):
            gid = int(all_v[i].split('_')[0])
            seqid = int(all_v[i].split('_')[1])
            v_dict[GRAPH.save_numbers(gid,seqid,16,32)]=i
            namedict[i] = self.id2v_dict[all_v[i]]
            namelist.append(self.id2v_dict[all_v[i]])            

        ali=Manager().dict()
        for v in range(self.Match_num+1):
            ali[v] = 0
        q = Queue()
        delta_head_dict=init_delta_head()
        startnodelist = list(self.DAG.startnodeset_clique)
        endnodes=[]
        for node in self.graphEndNodes:
            endnodes.append(node)
        v_num = len(all_v)
        for i in range(v_num):
            gid = int(all_v[i].split('_')[0])
            seqid = int(all_v[i].split('_')[1])
            v_dict[GRAPH.save_numbers(gid,seqid,16,32)]=i

        processlist=[]
        pool_num = self.pool_num
        for idx in range(pool_num):
            processlist.append(Process(target=calculate_delta,args=(goon_flag,lock,startnodelist[idx::pool_num],delta_head_dict, )))
        processlist.append(Process(target=is_end_f,args=(goon_flag, )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        indegree_dict_shm.close()
        indegree_dict_shm.unlink()
        
        
        

        if not os.path.exists(self.Viterbi_DAG_Path+'ali_result_{}/'.format(self.parameterName)):
            os.makedirs(self.Viterbi_DAG_Path+'ali_result_{}/'.format(self.parameterName))
        hidden_states = Manager().dict()
        outdegreeDict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.clique_num*np.dtype(np.uint16).itemsize)
        outdegreeDict =np.ndarray((self.DAG.clique_num,), dtype=np.int16, buffer=outdegreeDict_shm.buf)
        outdegreeDict[:] = np.zeros(self.DAG.clique_num)
        doneList[:] = np.full(self.DAG.totalNodes,1)
        for link in self.clique_link:
            outdegreeDict[link[0]]+=1
        q = Queue()
        endnldelist = list(self.DAG.endnodeset_clique)
        print()
        goon_flag = Value('i',1)
        processlist=[]
        pool_num = max(int(self.pool_num/2),1)
        print('pool_num',pool_num)
        for idx in range(pool_num):
            processlist.append(Process(target=calculate_state,args=(goon_flag,lock,endnldelist[idx::pool_num], )))
        processlist.append(Process(target=is_end_b,args=(goon_flag, )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]

        np.save(self.Viterbi_DAG_Path+'ali_result_{}/hiddenstate.npy'.format(self.parameterName),dict(hidden_states))
        np.save(self.Viterbi_DAG_Path+'ali_result_{}/insert_length_dict.npy'.format(self.parameterName),dict(ali))

        np.save(self.Viterbi_DAG_Path+'ali_result_{}/namelist.npy'.format(self.parameterName),namelist)

        if mode == 'memory':
            delta_M_mem_shm.close()
            delta_M_mem_shm.unlink()
            delta_I_mem_shm.close()
            delta_I_mem_shm.unlink()
            delta_D_mem_shm.close()
            delta_D_mem_shm.unlink()
            laststate_mem_shm.close()
            laststate_mem_shm.unlink()
            orinode_mem_shm.close()
            orinode_mem_shm.unlink()

        outdegreeDict_shm.close()
        outdegreeDict_shm.unlink()
        doneList_shm.close()
        doneList_shm.unlink()

    
    def state_to_aligment(self,seqiddb,mode='local',save_fasta=False,matrix=False):
        def query_contend_sequence_id(node):
            return seqiddb.findSequenceSource(self.DAG.nodeList[node][3],self.DAG.firstBiteofONM,self.DAG.allBiteofONM,self.DAG.firstBiteofOSM,self.DAG.allBiteofOSM)

        def draw_ali(nodelist):
            # print(self.DAG.totalNodes)
            vectorized_v_dict = np.vectorize(v_dict.get)
            loacal_xdict = xdict.copy()
            loacal_ali = ali.copy()
            spnode_statedict={}
            for node in nodelist:
                
                aa = self.DAG.nodeList[node][1]

                node_hidden_states = hidden_states[node]
                for state in node_hidden_states.keys():
                    if state[0]==2:
                        if (2,state[1]) in loacal_xdict.keys():
                            sx = loacal_xdict[(2,state[1])] + (loacal_ali[state[1]]-node_hidden_states[state][1])
                        else:
                            spnode_statedict[node] = spnode_statedict.get(node,{})
                            spnode_statedict[node][state]=hidden_states[node][state]
                            continue
                    else:
                        sx = loacal_xdict[(0,state[1])]
                    if isinstance(node_hidden_states[state][0],str):
                        contentvirus = query_contend_sequence_id(node)
                    else:
                        contentvirus = node_hidden_states[state][0]
                    if contentvirus.size!=0:
                        content = vectorized_v_dict(contentvirus[0])
                        lock.acquire()
                        ali_matrix[sx,content] = self.allBaseDict.get(aa,14) + 1
                        lock.release()
        # exit()

        ali = np.load(self.Viterbi_DAG_Path+'ali_result_{}/insert_length_dict.npy'.format(self.parameterName),allow_pickle=True).item()

        namelist = np.load(self.Viterbi_DAG_Path+'ali_result_{}/namelist.npy'.format(self.parameterName))
        v_dict={}
        all_v = sorted(self.all_virus, key=lambda x: (int(x.split("_")[0]), int(x.split("_")[-1])))
        v_num = len(all_v)
        for i in range(v_num):
            gid = int(all_v[i].split('_')[0])
            seqid = int(all_v[i].split('_')[1])
            v_dict[GRAPH.save_numbers(gid,seqid,16,32)]=i

        lock=Lock()
        sdict ={0:'A',1:'T',2:'C',3:'G'}
        seqdict = {}
        seqdict['x']=[]
        seqdict['ref']=[]
        xdict={}
        alilength=0
        ss=0
        for v in range(self.Match_num+1):
            ss+=ali[v]
            xdict[(2,v)] = alilength
            for i in range(alilength,alilength+ali[v]):
                seqdict['x'].append([2,v])
                seqdict['ref'].append(' ')
                
            alilength+=ali[v]
            if v!= self.Match_num : 
                xdict[(0,v)] = alilength
                seqdict['x'].append([0,v])
                seqdict['ref'].append(sdict[np.argmax(self.Me_Matrix[v])])
                alilength+=1
        print(alilength)

        if not os.path.exists(self.Viterbi_DAG_Path+'Matrix/'):
            os.makedirs(self.Viterbi_DAG_Path+'Matrix/')

        ali_matrix_shm = shared_memory.SharedMemory(create=True, size=alilength*self.vnum*np.dtype(np.uint8).itemsize)
        ali_matrix =np.ndarray((alilength,self.vnum), dtype=np.uint8, buffer=ali_matrix_shm.buf)

        pool_num = 20
        hidden_states = np.load(self.Viterbi_DAG_Path+'ali_result_{}/hiddenstate.npy'.format(self.parameterName),allow_pickle=True).item()
        processlist =[]
        klist = list(hidden_states.keys())
        for idx in range(pool_num):
            # print(klist[idx::pool_num])
            processlist.append(Process(target=draw_ali,args=(klist[idx::pool_num], )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        # exit()
        # zip_ali = []
        # for i in range(alilength):
        #     tmplist=[]
        #     if seqdict['x'][i][0]==0:
        #         ali_array = ali_matrix[i]
        #         max_base = np.argmax(self.Me_Matrix[seqdict['x'][i][1]])+1
        #         for base in set(ali_array):
        #             if base==max_base:
        #                 tmplist.insert(0,max_base)
        #             else:
        #                 tmplist.append([base,np.where(ali_array == base)[0]])
        #         zip_ali.append(tmplist)
        #     else:
        #         ali_array = ali_matrix[i]
        #         tmplist.insert(0,0)
        #         for base in set(ali_array)-{0}:
        #             tmplist.append([base,np.where(ali_array == base)[0]])
        #         zip_ali.append(tmplist)
        # zip_ali.append(self.vnum)
        zip_ali = []
        for i in range(alilength):
            
            ali_array = ali_matrix[i]
            # print(ali_array)
            unique_elements, counts = np.unique(ali_array, return_counts=True)
            max_count_index = np.argmax(counts)
            most_common_element = unique_elements[max_count_index]
            elements_indices = [[element, np.where(ali_array == element)[0]] for element in unique_elements if element != most_common_element]
            elements_indices.insert(0,most_common_element)
            # print(elements_indices)
            zip_ali.append(elements_indices)

        zip_ali.append(self.vnum)   # #
        zip_ali = np.array(zip_ali,dtype=object)  # # 7 ? √
        # print('***啊啊啊777')
        # np.save(self.Viterbi_DAG_Path+'ali_result_{}/zipali.npy'.format(self.parameterName),zip_ali)
        np.save(self.Viterbi_DAG_Path+'ali_result_{}/zipali.npy'.format(self.parameterName),zip_ali)

        ali_matrix = ali_matrix.T

        if matrix==True:
            np.save(self.Viterbi_DAG_Path+'ali_result_{}/ali_matrix.npy'.format(self.parameterName),ali_matrix)
        np.save(self.Viterbi_DAG_Path+'ali_result_{}/indexdict.npy'.format(self.parameterName),xdict)
        np.save(self.Viterbi_DAG_Path+'ali_result_{}/seqdict.npy'.format(self.parameterName),seqdict)
        np.save(self.Viterbi_DAG_Path+'ali_result_{}/namelist.npy'.format(self.parameterName),namelist)
        print('alisaved')


        if save_fasta:
            ssdict = {0:'m',2:'i'}
            MorI=''
            ref_seq=''
            seqlist=[]
            for s in range(alilength):
                MorI+=ssdict[seqdict['x'][s][0]]
                if ssdict[seqdict['x'][s][0]]=='m':
                    ref_seq+=seqdict['ref'][s]
                else:
                    ref_seq+='-'
            
            vectorized_draw_dict = np.vectorize(self.alignmentBaseDictionary.get)
            string_matrix = vectorized_draw_dict(ali_matrix)
            print('start write')
            seqlist = [SeqRecord(Seq(''.join(i)),id=namelist[idx],description='') for idx,i in tqdm(enumerate(string_matrix))]
            fastaseq=SeqRecord(Seq(MorI),id='MorI',description='')
            seqlist.insert(0,fastaseq)
            fastaseq=SeqRecord(Seq(ref_seq),id='Ref_seq',description='')
            seqlist.insert(0,fastaseq)
            SeqIO.write(seqlist,self.Viterbi_DAG_Path+'ali_result_{}/'.format(self.parameterName)+'aliresult.fasta','fasta')
            print(self.Viterbi_DAG_Path+'ali_result_{}/'.format(self.parameterName)+'aliresult.fasta')

        ali_matrix_shm.close()
        ali_matrix_shm.unlink()

    def Viterbi_seq(self,fasta_path,print_path=''):
        def vtb_seq(name,seq,spwlength=500):

            length = len(seq)

            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            Match_num = self.Match_num
            delta_M_mem=np.full((length,Match_num),np.NINF,dtype=np.float64)
            delta_I_mem=np.full((length,Match_num+1),np.NINF,dtype=np.float64)
            delta_D_mem=np.full((length,Match_num),np.NINF,dtype=np.float64)
            laststate_mem=np.full((length,3*Match_num+1),2,dtype=np.int8)

            
            # delta_D_mem.append(np.array([np.NINF]*Match_num))
            tmp_array = np.full(Match_num,np.NINF,dtype=np.float64)
            tmp_array[0] = self.pi_D
            for i in range(1,Match_num):
                tmp_array[i] = tmp_array[i-1]+D2D_array[i-1]
            hidden_states = []
            
            baseID = self.allBaseDict[seq[0]]
            Me = self.Me_Matrix_degenerate_base[baseID]
            Ie = self.Ie_Matrix_degenerate_base[baseID]
            delta_M = np.full(Match_num, np.NINF, dtype=np.float64)
            delta_I = np.full(Match_num + 1, np.NINF, dtype=np.float64)
            delta_D = np.full(Match_num, np.NINF, dtype=np.float64)
            delta_M[0]=self.pi_M+Me[0]
            delta_I[0]=self.pi_I+Ie[0]
            last_delta_D = tmp_array
            delta_M[1:] = last_delta_D[:-1]+D2M_array+Me[1:]
            delta_I[1:] = last_delta_D+D2I_array+Ie[1:]
            
            maxProbOrigin_D = np.array([0]*(Match_num),dtype='int')
            maxProbOrigin_D[0]=2
            delta_D[0]=delta_I[0]+I2D_array[0]

            for i in range(1,Match_num):
                D_arg = [delta_M[i-1]+M2D_array[i-1],delta_D[i-1]+D2D_array[i-1],delta_I[i]+I2D_array[i]]
                delta_D[i] = np.max(D_arg)
                maxProbOrigin_D[i] = np.argmax(D_arg)

            maxProbOrigin_M = np.full(Match_num, 2, dtype='int')
            maxProbOrigin_M[0]=3
            maxProbOrigin_I = np.full(Match_num+1, 2, dtype='int')
            maxProbOrigin_I[0]=3
            delta_M_mem[0]=delta_M
            delta_I_mem[0]=delta_I
            delta_D_mem[0]=delta_D
            laststate_mem[0]=np.concatenate((maxProbOrigin_M,maxProbOrigin_D,maxProbOrigin_I))

            for i in tqdm(range(1,length)):
                baseID = self.allBaseDict[seq[i]]
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]

                last_delta_M = delta_M_mem[i-1]
                last_delta_I = delta_I_mem[i-1]
                last_delta_D = delta_D_mem[i-1]
                delta_I, delta_M, delta_D, maxProbOrigin_I, maxProbOrigin_M, maxProbOrigin_D = Profile_HMM.calculate_delta_values(0, Match_num, Match_num,
                    last_delta_I, last_delta_M, last_delta_D,
                    I2I_array, I2M_array, M2I_array, M2M_array, D2I_array, D2M_array, I2D_array, D2D_array,
                    Ie, Me,M2D_array)

                delta_M_mem[i]=delta_M
                delta_I_mem[i]=delta_I
                delta_D_mem[i]=delta_D

                laststate_mem[i]=np.concatenate((maxProbOrigin_M,maxProbOrigin_D,maxProbOrigin_I))

            decrease_list=[-1,-1,0]
            delta_all = [delta_M_mem[-1],delta_D_mem[-1],delta_I_mem[-1]]

            # 尾端节点
            nowstate = np.argmax([delta_all[0][Match_num-1]+self.M2E,delta_all[1][Match_num-1]+self.D2E,delta_all[2][Match_num]+self.I2E])
            nowstateindex = Match_num+decrease_list[nowstate]
            while nowstate==1:
                nowstate = laststate_mem[-1][nowstate*Match_num+nowstateindex]
                nowstateindex += decrease_list[nowstate]
            if nowstate==2:
                Inum=1
            else:
                Inum=0
            hidden_states.append([nowstate,nowstateindex,Inum,seq[-1]])

            for i in range(length-1)[::-1]:
                oristateindex = hidden_states[0][1]
                ori_Inum = hidden_states[0][2]
                nowstate = hidden_states[0][0]
                nowstateindex = hidden_states[0][1]

                tmpnowstate = laststate_mem[i+1][nowstate*Match_num+nowstateindex]
                nowstate = tmpnowstate
                nowstateindex += decrease_list[tmpnowstate]
                while nowstate==1:
                    nowstate = laststate_mem[i][nowstate*Match_num+nowstateindex]
                    nowstateindex += decrease_list[nowstate]

                if nowstate==2 and nowstateindex == oristateindex:
                    hidden_states.insert(0,[nowstate,nowstateindex,ori_Inum+1,seq[i]])
                    ali[nowstateindex] = max([ori_Inum+1,ali[nowstateindex]])
                elif nowstate==2:
                    hidden_states.insert(0,[nowstate,nowstateindex,1,seq[i]])
                    ali[nowstateindex] = max([ori_Inum+1,ali[nowstateindex]])
                else:
                    hidden_states.insert(0,[nowstate,nowstateindex,0,seq[i]])

            statedict[name]=hidden_states

            
            
        def paralla_draw(lst):
            for i in tqdm(lst):
                vtb_seq(i[0],i[1])



        
        ali=Manager().dict()
        for v in range(self.Match_num+1):
            ali[v] = 0
        statedict= Manager().dict()
        Sequence_record_list = SeqIO.parse(fasta_path, "fasta")
        sequence_list=[]
        for record in Sequence_record_list:
            sequence_list.append([record.id,record.seq])

        pool_num = 2


        processlist =[]
        for index in range(pool_num):

            processlist.append(Process(target=paralla_draw,args=(sequence_list[index::pool_num], )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]

        sdict ={0:'A',1:'T',2:'C',3:'G'}
        seqdict = {}
        seqdict['x']=[]
        seqdict['ref']=[]
        xdict={}
        alilength=0
        ss=0
        for v in range(self.Match_num+1):
            ss+=ali[v]
            xdict[(2,v)] = alilength
            for i in range(alilength,alilength+ali[v]):
                seqdict['x'].append([2,v])
                seqdict['ref'].append(' ')
                
            alilength+=ali[v]
            if v!= self.Match_num : 
                xdict[(0,v)] = alilength
                seqdict['x'].append([0,v])
                seqdict['ref'].append(sdict[np.argmax(self.Me_Matrix[v])])
                alilength+=1

        ali_matrix_shm = shared_memory.SharedMemory(create=True, size=alilength*self.vnum*np.dtype(np.uint8).itemsize)
        ali_matrix =np.ndarray((self.vnum,alilength), dtype=np.uint8, buffer=ali_matrix_shm.buf)
        index = 0
        for seq_id_and_seq in tqdm(sequence_list):
            # seq = '-'*alilength
            if seq_id_and_seq[1]!='':
                seq_id = seq_id_and_seq[0]
                for state in statedict[seq_id]:
                    if state[0]==2:
                        x = xdict[(2,state[1])] + (ali[state[1]]-state[2])
                    else:
                        x = xdict[(0,state[1])]

                    ali_matrix[index,x]=self.allBaseDict[state[3]] + 1
            index+=1

        np.save(self.train_DAG_Path+'ali_matrix_single_seq.npy',ali_matrix)
        if print_path=='':
            name = self.train_DAG_Path+'result.fasta'
        else:
            name = print_path
        print('save ali in '+name)
        ssdict = {0:'m',2:'i'}
        MorI=''
        ref_seq=''
        seqlist=[]
        for s in range(alilength):
            MorI+=ssdict[seqdict['x'][s][0]]
            if ssdict[seqdict['x'][s][0]]=='m':
                ref_seq+=seqdict['ref'][s]
            else:
                ref_seq+='-'
        vectorized_draw_dict = np.vectorize(self.alignmentBaseDictionary.get)
        string_matrix = vectorized_draw_dict(ali_matrix)
        print('start write')
        seqlist = [SeqRecord(Seq(''.join(i)),id=self.vlist[idx],description='') for idx,i in tqdm(enumerate(string_matrix))]
        fastaseq=SeqRecord(Seq(MorI),id='MorI',description='')
        seqlist.insert(0,fastaseq)
        fastaseq=SeqRecord(Seq(ref_seq),id='Ref_seq',description='')
        seqlist.insert(0,fastaseq)
        SeqIO.write(seqlist,name,'fasta')
        print(name)
        ali_matrix_shm.close()
        ali_matrix_shm.unlink()
    
    def Viterbi_seq_windows(self,fasta_path,print_path=''):
        def vtb_seq(name,seq,ref_seq,wlength=300):
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            Match_num = self.Match_num

            length = len(seq)
            range_list = GRAPH.pair_range(str(seq),str(ref_seq),self.DAG.originalfragmentLength)

            range_length=0
            left_right_list=[]
            start_and_end_index_list = []
            for i in range_list:
                left_right_list.append(max(i[0]-wlength,0))
                left_right_list.append(min(i[1]+wlength,Match_num))
                sted=[range_length]
                range_length+=(left_right_list[-1]-left_right_list[-2]+1)
                sted.append(range_length)
                start_and_end_index_list.append(sted)

            delta_M_mem=np.full((range_length),np.NINF,dtype=np.float64)
            delta_I_mem=np.full((range_length),np.NINF,dtype=np.float64)
            delta_D_mem=np.full((range_length),np.NINF,dtype=np.float64)
            laststate_mem=np.full((3*range_length),2,dtype=np.int8)
            
            tmp_delta_D = np.array([np.NINF]*Match_num)
            tmp_delta_D[0] = self.pi_D
            for i in range(1,Match_num):
                tmp_delta_D[i] = tmp_delta_D[i-1]+D2D_array[i-1]
            hidden_states = []
            
            i=0
            stateRangeStart,stateRangeEnd = left_right_list[2*i:2*i+2]
            arrayRangeStart,arrayRangeEnd = start_and_end_index_list[i]
            array_length = stateRangeEnd-stateRangeStart

            baseID = self.allBaseDict[seq[0]]
            Me = self.Me_Matrix_degenerate_base[baseID]
            Ie = self.Ie_Matrix_degenerate_base[baseID]
            delta_M = np.full(array_length, np.NINF, dtype=np.float64)
            delta_I = np.full(array_length + 1, np.NINF, dtype=np.float64)
            delta_D = np.full(array_length, np.NINF, dtype=np.float64)
            delta_M[0]=self.pi_M+Me[0]
            delta_I[0]=self.pi_I+Ie[0]
            last_delta_D = tmp_delta_D[stateRangeStart:stateRangeEnd]
            delta_M[1:] = last_delta_D[:-1]+D2M_array[stateRangeStart:stateRangeEnd-1]+Me[stateRangeStart+1:stateRangeEnd]
            delta_I[1:] = last_delta_D+D2I_array[stateRangeStart:stateRangeEnd]+Ie[stateRangeStart+1:stateRangeEnd+1]

            maxProbOrigin_D = np.full(array_length, 2, dtype='int')
            delta_D[0]=delta_I[0]+I2D_array[0]

            for i in range(1,array_length):
                D_arg = [delta_M[i-1]+M2D_array[stateRangeStart+i-1],delta_D[i-1]+D2D_array[stateRangeStart+i-1],delta_I[i]+I2D_array[stateRangeStart+i]]
                delta_D[i] = np.max(D_arg)
                maxProbOrigin_D[i] = np.argmax(D_arg)

            maxProbOrigin_M = np.full(array_length, 2, dtype='int')
            maxProbOrigin_I = np.full(array_length+1, 2, dtype='int')
            delta_M_mem[arrayRangeStart:arrayRangeEnd-1]=delta_M
            delta_I_mem[arrayRangeStart:arrayRangeEnd]=delta_I
            delta_D_mem[arrayRangeStart:arrayRangeEnd-1]=delta_D
            st=3*arrayRangeStart
            laststate_mem[st:st+array_length]=maxProbOrigin_M
            st+=array_length
            laststate_mem[st:st+array_length]=maxProbOrigin_D
            st+=array_length
            laststate_mem[st:st+array_length+1]=maxProbOrigin_I


            for i in range(1,length):
                stateRangeStart,stateRangeEnd = left_right_list[2*i:2*i+2]
                f_node = i-1
                f_left_limit,f_right_limit = left_right_list[2*f_node:2*f_node+2]
                arrayRangeStart,arrayRangeEnd = start_and_end_index_list[i]
                _arrayRangeStart,_arrayRangeEnd = start_and_end_index_list[f_node]
                array_length = stateRangeEnd-stateRangeStart

                baseID = self.allBaseDict[seq[i]]
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                last_delta_M=np.full(Match_num,np.NINF)
                last_delta_I=np.full(Match_num+1,np.NINF)
                last_delta_D=np.full(Match_num,np.NINF)

                last_delta_M[f_left_limit:f_right_limit] = delta_M_mem[_arrayRangeStart:_arrayRangeEnd-1]
                last_delta_I[f_left_limit:f_right_limit+1] = delta_I_mem[_arrayRangeStart:_arrayRangeEnd]
                last_delta_D[f_left_limit:f_right_limit] = delta_D_mem[_arrayRangeStart:_arrayRangeEnd-1]
                last_delta_M = last_delta_M[stateRangeStart:stateRangeEnd]
                last_delta_I = last_delta_I[stateRangeStart:stateRangeEnd+1]
                last_delta_D = last_delta_D[stateRangeStart:stateRangeEnd]

                delta_I, delta_M, delta_D, maxProbOrigin_I, maxProbOrigin_M, maxProbOrigin_D = Profile_HMM.calculate_delta_values(stateRangeStart, stateRangeEnd, array_length,
                                                                        last_delta_I, last_delta_M, last_delta_D,
                                                                        I2I_array, I2M_array, M2I_array, M2M_array, D2I_array, D2M_array, I2D_array, D2D_array,
                                                                        Ie, Me,M2D_array)

                delta_M_mem[arrayRangeStart:arrayRangeEnd-1]=delta_M
                delta_I_mem[arrayRangeStart:arrayRangeEnd]=delta_I
                delta_D_mem[arrayRangeStart:arrayRangeEnd-1]=delta_D
                
                st=3*arrayRangeStart
                laststate_mem[st:st+array_length]=maxProbOrigin_M
                st+=array_length
                laststate_mem[st:st+array_length]=maxProbOrigin_D
                st+=array_length
                laststate_mem[st:st+array_length+1]=maxProbOrigin_I



            i= length-1
            stateRangeStart,stateRangeEnd = left_right_list[2*i:2*i+2]
            arrayRangeStart,arrayRangeEnd = start_and_end_index_list[i]
            array_length = stateRangeEnd-stateRangeStart
            decrease_list=[-1,-1,0]
            delta_all = [delta_M_mem[arrayRangeEnd-2],delta_D_mem[arrayRangeEnd-2],delta_I_mem[arrayRangeEnd-1]]

            # 尾端节点
            laststate_local = laststate_mem[3*arrayRangeStart:3*arrayRangeEnd]
            nowstate = np.argmax([delta_all[0]+self.M2E,delta_all[1]+self.D2E,delta_all[2]+self.I2E])
            nowstateindex = Match_num+decrease_list[nowstate]
            while nowstate==1:
                nowstate = laststate_local[nowstate*array_length+nowstateindex-stateRangeStart]
                nowstateindex += decrease_list[nowstate]
            if nowstate==2:
                Inum=1
            else:
                Inum=0
            hidden_states.append([nowstate,nowstateindex,Inum,seq[-1]])

            for i in range(length-1)[::-1]:

                stateRangeStart,stateRangeEnd = left_right_list[2*i:2*i+2]
                snode=i+1
                s_left_limit,s_right_limit = left_right_list[2*snode:2*snode+2]
                arrayRangeStart,arrayRangeEnd = start_and_end_index_list[i]
                s_Matrix_start_point,s_Matrix_end_point = start_and_end_index_list[snode]
                laststate_local = laststate_mem[3*s_Matrix_start_point:3*s_Matrix_end_point]
                state_local = laststate_mem[3*arrayRangeStart:3*arrayRangeEnd]
                array_length = stateRangeEnd-stateRangeStart
                s_array_length = s_right_limit-s_left_limit

                oristateindex = hidden_states[0][1]
                ori_Inum = hidden_states[0][2]
                nowstate = hidden_states[0][0]
                nowstateindex = hidden_states[0][1]

                tmpnowstate = laststate_local[nowstate*s_array_length+(nowstateindex-s_left_limit)]
                
                nowstate = tmpnowstate
                nowstateindex += decrease_list[tmpnowstate]

                nowstate, nowstateindex = Profile_HMM.update_state(state_local,array_length,nowstate, nowstateindex, stateRangeStart, decrease_list)

                if nowstate==2 and nowstateindex == oristateindex:
                    hidden_states.insert(0,[nowstate,nowstateindex,ori_Inum+1,seq[i]])
                    ali[nowstateindex] = max([ori_Inum+1,ali[nowstateindex]])
                elif nowstate==2:
                    hidden_states.insert(0,[nowstate,nowstateindex,1,seq[i]])
                    ali[nowstateindex] = max([ori_Inum+1,ali[nowstateindex]])
                else:
                    hidden_states.insert(0,[nowstate,nowstateindex,0,seq[i]])
                    
                    
            statedict[name]=hidden_states

            
            
        def paralla_draw(lst):
            for i in tqdm(lst):
                vtb_seq(i[0],i[1],self.ref_seq)



        
        ali=Manager().dict()
        for v in range(self.Match_num+1):
            ali[v] = 0
        statedict= Manager().dict()
        Sequence_record_list = SeqIO.parse(fasta_path, "fasta")
        sequence_list=[]
        for record in Sequence_record_list:
            sequence_list.append([record.id,record.seq])
        pool_num = 5



        processlist =[]
        for index in range(pool_num):

            processlist.append(Process(target=paralla_draw,args=(sequence_list[index::pool_num], )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]

        sdict ={0:'A',1:'T',2:'C',3:'G'}
        seqdict = {}
        seqdict['x']=[]
        seqdict['ref']=[]
        xdict={}
        alilength=0
        ss=0
        for v in range(self.Match_num+1):
            ss+=ali[v]
            xdict[(2,v)] = alilength
            for i in range(alilength,alilength+ali[v]):
                seqdict['x'].append([2,v])
                seqdict['ref'].append(' ')
                
            alilength+=ali[v]
            if v!= self.Match_num : 
                xdict[(0,v)] = alilength
                seqdict['x'].append([0,v])
                seqdict['ref'].append(sdict[np.argmax(self.Me_Matrix[v])])
                alilength+=1

        ali_matrix_shm = shared_memory.SharedMemory(create=True, size=alilength*self.vnum*np.dtype(np.uint8).itemsize)
        ali_matrix =np.ndarray((self.vnum,alilength), dtype=np.uint8, buffer=ali_matrix_shm.buf)
        index = 0
        for seq_id_and_seq in tqdm(sequence_list):
            if seq_id_and_seq[1]!='':
                seq_id = seq_id_and_seq[0]
                for state in statedict[seq_id]:
                    if state[0]==2:
                        x = xdict[(2,state[1])] + (ali[state[1]]-state[2])
                    else:
                        x = xdict[(0,state[1])]

                    ali_matrix[index,x]=self.allBaseDict[state[3]] + 1
            index+=1

        
        if print_path=='':
            print_path = self.train_DAG_Path
        
        
        np.save(print_path+'ali_matrix_single_seq.npy',ali_matrix)
        print('save ali in '+print_path)
        ssdict = {0:'m',2:'i'}
        MorI=''
        ref_seq=''
        seqlist=[]
        for s in range(alilength):
            MorI+=ssdict[seqdict['x'][s][0]]
            if ssdict[seqdict['x'][s][0]]=='m':
                ref_seq+=seqdict['ref'][s]
            else:
                ref_seq+='-'
        vectorized_draw_dict = np.vectorize(self.alignmentBaseDictionary.get)
        string_matrix = vectorized_draw_dict(ali_matrix)
        print('start write')
        seqlist = [SeqRecord(Seq(''.join(i)),id=self.vlist[idx],description='') for idx,i in tqdm(enumerate(string_matrix))]
        fastaseq=SeqRecord(Seq(MorI),id='MorI',description='')
        seqlist.insert(0,fastaseq)
        fastaseq=SeqRecord(Seq(ref_seq),id='Ref_seq',description='')
        seqlist.insert(0,fastaseq)
        SeqIO.write(seqlist,print_path+'single_seq.fasta','fasta')
        print(print_path+'single_seq.fasta')
        ali_matrix_shm.close()
        ali_matrix_shm.unlink()
    def save_fasta(gp_path,pc_name,save_path,Ref=False):

        draw_dict = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}
        print(gp_path,pc_name)
        ali_matrix = np.load(gp_path+'ali_result_{}/ali_matrix.npy'.format(pc_name))
        seqdict = np.load(gp_path+'ali_result_{}/seqdict.npy'.format(pc_name),allow_pickle=True).item()
        namelist = np.load(gp_path+'ali_result_{}/namelist.npy'.format(pc_name))

        alilength = ali_matrix.shape[1]

        ssdict = {0:'m',2:'i'}
        MorI=''
        ref_seq=''
        seqlist=[]
        if Ref:
            for s in range(alilength):
                MorI+=ssdict[seqdict['x'][s][0]]
                if ssdict[seqdict['x'][s][0]]=='m':
                    ref_seq+=seqdict['ref'][s]
                else:
                    ref_seq+='-'
        
        vectorized_draw_dict = np.vectorize(draw_dict.get)
        string_matrix = vectorized_draw_dict(ali_matrix)
        print('start write')
        seqlist = [SeqRecord(Seq(''.join(i)),id=namelist[idx],description='') for idx,i in tqdm(enumerate(string_matrix))]
        fastaseq=SeqRecord(Seq(MorI),id='MorI',description='')
        seqlist.insert(0,fastaseq)
        fastaseq=SeqRecord(Seq(ref_seq),id='Ref_seq',description='')
        seqlist.insert(0,fastaseq)
        SeqIO.write(seqlist,save_path,'fasta')
        print(save_path)
        pass

    def save_fasta_zip(gp_path,pc_name,save_path,Ref=False):
        
        draw_dict = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}
        zipali = np.load(gp_path+'ali_result_{}/zipali.npy'.format(pc_name),allow_pickle=True)
        seqdict = np.load(gp_path+'ali_result_{}/seqdict.npy'.format(pc_name),allow_pickle=True).item()
        namelist = np.load(gp_path+'ali_result_{}/namelist.npy'.format(pc_name))
        ali_matrix = np.full((len(namelist),len(zipali)),0)
        print((len(namelist),len(zipali)))
        for index,ali in enumerate(zipali):
            # print(ali)
            # print(index)
            ali_matrix[:,index] = ali[0]
            for base in ali[1:]:
                try:
                    ali_matrix[base[1],index] = base[0]
                except:
                    print(base[1],index,base[0])
                    exit()
        alilength = ali_matrix.shape[1]

        ssdict = {0:'m',2:'i'}
        MorI=''
        ref_seq=''
        seqlist=[]
        if Ref:
            for s in range(alilength):
                MorI+=ssdict[seqdict['x'][s][0]]
                if ssdict[seqdict['x'][s][0]]=='m':
                    ref_seq+=seqdict['ref'][s]
                else:
                    ref_seq+='-'
        
        vectorized_draw_dict = np.vectorize(draw_dict.get)
        string_matrix = vectorized_draw_dict(ali_matrix)
        print('start write')
        seqlist = [SeqRecord(Seq(''.join(i)),id=namelist[idx],description='') for idx,i in tqdm(enumerate(string_matrix))]
        fastaseq=SeqRecord(Seq(MorI),id='MorI',description='')
        seqlist.insert(0,fastaseq)
        fastaseq=SeqRecord(Seq(ref_seq),id='Ref_seq',description='')
        seqlist.insert(0,fastaseq)
        SeqIO.write(seqlist,save_path+'result.fasta','fasta')
        print(save_path)

def build_adb_merge(osmpath,dbpath,lst):
    if os.path.exists(dbpath+"osm.db"):
        os.remove(dbpath+"osm.db")
    conn = sqlite3.connect(dbpath+"osm.db")
    
    cursor = conn.cursor()
    for setname in lst:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vset{} (
            node_id INTEGER PRIMARY KEY,
            virus BLOB NOT NULL
        )
        '''.format(setname))

        conn.commit()
        print(osmpath+str(setname)+'/osm.npy')
        arrays = np.load(osmpath+str(setname)+'/osm.npy',allow_pickle=True)
        binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]
        df = pandas.DataFrame(binary_arrays, columns=['node_id','virus'])

        df.to_sql('vset'+str(setname), con=conn, if_exists='append', index=False)

        conn.commit()
        sql = '''CREATE INDEX nodeid{0} ON vset{0} (node_id);'''.format(setname)
        cursor.execute(sql)
        conn.commit()
        
    conn.close()
def build_adb(path,setname):
    if os.path.exists(path+"osm.db"):
        os.remove(path+"osm.db")
    conn = sqlite3.connect(path+"osm.db")
    
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vset{} (
        node_id INTEGER PRIMARY KEY,
        virus BLOB NOT NULL
    )
    '''.format(setname))

    conn.commit()
    print(setname)
    arrays = np.load(path+'/osm.npy',allow_pickle=True)
    binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]
    df = pandas.DataFrame(binary_arrays, columns=['node_id','virus'])

    df.to_sql('vset'+str(setname), con=conn, if_exists='append', index=False)

    conn.commit()
    sql = '''CREATE INDEX nodeid{0} ON vset{0} (node_id);'''.format(setname)
    cursor.execute(sql)
    conn.commit()
    conn.close()

def AandZ(inpath,savepath,newfranum):
    graph = load_graph(inpath)
    for node in graph.nodeList:
        node[3] = [(0,node[0])]
    startseqs=set()
    for node in graph.startNodeSet:
        startseqs.add(graph.nodeList[node][1])
    newfra = newfranum
    graph.merge_check(mode='ali')
    if newfra<graph.fragmentLength:
        graph.fragmentReduce(newfra) # 图原子化
        graph.mergeNodesWithSameCoordinates(newfra) 
    graph.save_graph(mode='zip',savepath=savepath)
    Trace_zip(inpath,savepath)
    # print()
    os.system('cp {}v_id.npy {}v_id.npy'.format(inpath,savepath))

def ini_pc(Ref_seq,emProbMatrix,ME,MD,MI,II,DM,pi_MID,outpath,parameter_name):
    Match_length = len(Ref_seq)
    _mi = np.full(Match_length+1,MI,dtype='float64') 
    _md = np.full(Match_length+1,MD,dtype='float64')
    _mm = np.full(Match_length+1,np.log(1-np.exp(MD)-np.exp(MI)),dtype='float64') #M+1
    _md[0] = np.log(pi_MID[2]/np.sum(pi_MID))
    _mi[0] = np.log(pi_MID[1]/np.sum(pi_MID))
    _mm[0] = np.log(pi_MID[0]/np.sum(pi_MID))
    _mm[-1]= ME
    _mi[-1]= np.log(1-np.exp(ME))
    _ii = np.full(Match_length+1,II,dtype='float64')
    _im = np.log(np.full(Match_length+1,1)-np.exp(_ii),dtype='float64')
    _id = np.full(Match_length+1,np.NINF,dtype='float64')
    _di =  np.full(Match_length+1,np.NINF,dtype='float64')
    _dm = np.full(Match_length+1,DM,dtype='float64')
    _dm[0] = np.NINF
    _dm[-1] = 0
    _dd = np.full(Match_length+1,np.log(1-np.exp(DM)),dtype='float64')
    _dd[0] = np.NINF
    _dd[-1] = np.NINF
    _em = np.log(emProbMatrix.T)
    _ei = np.full((_em.shape[0]+1,_em.shape[1]),np.log(1/4))
    parameterDict={"_mm": _mm,"_md": _md,"_mi": _mi,"_im": _im,"_id": _id,"_ii": _ii,"_dm": _dm,"_dd": _dd,"_di": _di,"match_emission":_em,"insert_emission":_ei}
    print(outpath+'ini/init_{}.npy'.format(parameter_name))
    np.save(outpath+'ini/init_{}.npy'.format(parameter_name),parameterDict)

def ini_pc_with_random(Ref_seq,emProbMatrix,ME,MD,MI,II,DM,pi_MID,outpath,parameter_name):
    Match_num = len(Ref_seq)
    _mi = np.full(Match_num+1,MI,dtype='float64') 
    _md = np.full(Match_num+1,MD,dtype='float64')
    _mm = np.full(Match_num+1,np.log(1-np.exp(MD)-np.exp(MI)),dtype='float64')
    # # Match状态出发转移概率加入随机 # # 8 替换，记得log一下
    for i in range(Match_num - 1):
        # print('开始：', _mm[i], _md[i], _mi[i])
        # print('***啊啊啊888')
        random_1 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))  #
        random_2 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))  #
        random_3 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))  #
        tmp = np.zeros(3,dtype=np.float64)             # 这里有几个tmp就写几，三个就是3，两个就是2
        tmp[0] = np.logaddexp(_mm[i],random_1)
        tmp[1] = np.logaddexp(_md[i],random_2)         # 注意这里面mmmdmi的顺序要和tmp012一致
        tmp[2] = np.logaddexp(_mi[i],random_3)
        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        _mm[i] = tmp[0] - Psum
        _md[i] = tmp[1] - Psum
        _mi[i] = tmp[2] - Psum
        # print('归一：', _mm[i], _md[i], _mi[i])

    # print('开始：', _mm[0], _md[0], _mi[0])
    _md[0] = np.log(pi_MID[2]/np.sum(pi_MID))
    _mi[0] = np.log(pi_MID[1]/np.sum(pi_MID))
    _mm[0] = np.log(pi_MID[0]/np.sum(pi_MID))
    # # Start状态出发的转移概率加入随机值 # # 9 替换，记得log一下
    # print('***啊啊啊999')
    random_1 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
    random_2 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
    random_3 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
    tmp = np.zeros(3,dtype=np.float64)
    tmp[0] = np.logaddexp(_mm[0],random_1)
    tmp[1] = np.logaddexp(_md[0],random_2)
    tmp[2] = np.logaddexp(_mi[0],random_3)
    # 归一化
    Psum = np.logaddexp.reduce(tmp,axis=0)
    _mm[0] = tmp[0] - Psum
    _md[0] = tmp[1] - Psum
    _mi[0] = tmp[2] - Psum
    # print('归一：', _mm[0], _md[0], _mi[0])

    # print('开始：', _mm[-1], _mi[-1])
    _mm[-1]= ME
    _mi[-1]= np.log(1-np.exp(ME))
    # # Start状态出发的转移概率加入随机值 # # 10 替换，记得log一下
    # print('***啊啊啊10')
    random_1 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
    random_2 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
    tmp = np.zeros(2,dtype=np.float64)
    tmp[0] = np.logaddexp(_mm[-1],random_1)
    tmp[1] = np.logaddexp(_mi[-1],random_2)
    # 归一化
    Psum = np.logaddexp.reduce(tmp,axis=0)
    _mm[-1] = tmp[0] - Psum
    _mi[-1] = tmp[1] - Psum
    # print('归一：', _mm[-1], _mi[-1])

    _ii = np.full(Match_num+1,II,dtype='float64')
    _im = np.log(np.full(Match_num+1,1)-np.exp(_ii),dtype='float64')
    for i in range(Match_num+1):
        # print('开始：', _ii[i], _im[i])
        # # 11 替换，记得log一下
        # print('***啊啊啊111111')
        random_1 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
        random_2 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
        tmp = np.zeros(2,dtype=np.float64)
        tmp[0] = np.logaddexp(_ii[i],random_1)
        tmp[1] = np.logaddexp(_im[i],random_2)
        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        _ii[i] = tmp[0] - Psum
        _im[i] = tmp[1] - Psum
        # print('归一：', _ii[i], _im[i])

    _id = np.full(Match_num+1,np.NINF,dtype='float64')  # id=0不用变，放到外面不容易误伤


    _di = np.full(Match_num+1,np.NINF,dtype='float64') # di其实也是0，放到外面不容易误伤
    _dm = np.full(Match_num+1,DM,dtype='float64')
    _dd = np.full(Match_num+1,np.log(1-np.exp(DM)),dtype='float64')  # dd非0，归一化非0的都不能漏掉
    for i in range(Match_num+1):
        # print('开始：', _dm[i], _dd[i])
        # # 12 替换，记得log一下
        # print('***啊啊啊12')
        random_1 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
        random_2 = np.log(np.random.uniform(i_global_random_low, i_global_random_up, 1))
        tmp = np.zeros(2,dtype=np.float64)
        tmp[0] = np.logaddexp(_dm[i],random_1)
        tmp[1] = np.logaddexp(_dd[i],random_2)
        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        _dm[i] = tmp[0] - Psum
        _dd[i] = tmp[1] - Psum
        # print('归一：', _dm[i], _dd[i])

    _dm[0] = np.NINF
    _dm[-1] = 0

    _dd[0] = np.NINF
    _dd[-1] = np.NINF

    _em = np.log(emProbMatrix.T)
    for i in range(Match_num):
        # print('开始：', _em[i])
        # # 13 此处替换为shape为self.normalBaseCount的随机array即可，记得log一下
        # print('***啊啊啊13')
        # random_array = np.log(0)   # # 注意，如果你想随机数置零，此处要这样修改，或者直接注释掉这里的整个for循环
        random_array = np.log(np.random.rand(len({"A":0,"T":1,"C":2,"G":3})))

        # 加入随机
        tmp = np.logaddexp(_em[i],random_array)
        # 归一化
        Psum = np.logaddexp.reduce(tmp,axis=0)
        _em[i] = tmp-Psum
        # print('归一：', _em[i])

    _ei = np.full((_em.shape[0]+1,_em.shape[1]),np.log(1/4))
    parameterDict={"_mm": _mm,"_md": _md,"_mi": _mi,"_im": _im,"_id": _id,"_ii": _ii,"_dm": _dm,"_dd": _dd,"_di": _di,"match_emission":_em,"insert_emission":_ei}
    print(outpath+'ini/init_{}.npy'.format(parameter_name))
    np.save(outpath+'ini/init_{}.npy'.format(parameter_name),parameterDict)

def ref_graph_build_local(graph_path,thr=0.001,MissMatchScore=-5):
    ref_dict = np.load(graph_path+'thr_{}.npz'.format(thr))
    ref_seq = str(ref_dict['ref_seq'])
    ref_node_list = list(ref_dict['ref_node_list'])
    emProbMatrix = ref_dict['emProbMatrix']

    emProbMatrix+=np.exp(MissMatchScore)
    sum_of_emProbMatrix = np.sum(emProbMatrix,axis=0)
    emProbMatrix = emProbMatrix/sum_of_emProbMatrix

    sum_of_emProbMatrix = np.sum(emProbMatrix,axis=0)
    emProbMatrix = emProbMatrix/sum_of_emProbMatrix
    pi_MID=[1,1,1]
    return ref_seq,ref_node_list,emProbMatrix,pi_MID

def ali(train_DAG_Path,Viterbi_DAG_Path,seqiddb,modify_dict,parameter_name,outpath):

    windows_length=100
    ref_seq,ref_node_list,emProbMatrix,pi_MID = ref_graph_build_local(outpath,thr=modify_dict['weight_thr'],MissMatchScore=modify_dict['emProbAdds_Match'])
    # # 14 ini_pc 原。   改：ini_pc_with_random
    # print('调用了吗？ali')  # 调用了
    ini_pc(ref_seq,emProbMatrix,modify_dict['init_D2End'],modify_dict['init_M2D'],modify_dict['init_M2I'],modify_dict['init_I2I'],modify_dict['init_D2M'],pi_MID,train_DAG_Path,parameter_name)
    parameter_path=train_DAG_Path+"ini/init_{}.npy".format(parameter_name)
    
    ph = Profile_HMM(train_DAG_Path,train_DAG_Path,parameter_name,parameter_path=parameter_path)
    ph.fit(train_DAG_Path,ref_node_list,ref_seq,modify_dict,True,windows_length)

    ph.init_viterbi_data(Viterbi_DAG_Path,ref_node_list,ref_seq,windows_length=windows_length)

    ph.Viterbi(seqiddb)
    ph.state_to_aligment(seqiddb,matrix=True)

def train(outpath,train_DAG_Path,parameter_name,modify_dict):
    windows_length=100
    ref_seq,ref_node_list,emProbMatrix,pi_MID = ref_graph_build_local(outpath,thr=modify_dict['weight_thr'],MissMatchScore=modify_dict['emProbAdds_Match'])
    # ini_pc_with_random
    # print('调用了吗？train')  # 没调用
    ini_pc(ref_seq,emProbMatrix,modify_dict['init_D2End'],modify_dict['init_M2D'],modify_dict['init_M2I'],modify_dict['init_I2I'],modify_dict['init_D2M'],pi_MID,train_DAG_Path,parameter_name)
    parameter_path=train_DAG_Path+"ini/init_{}.npy".format(parameter_name)
    # parameter_path=train_DAG_Path+"ini/{}_pc_8.npy".format(parameter_name)
    
    ph = Profile_HMM(train_DAG_Path,train_DAG_Path,parameter_name,parameter_path=parameter_path)
    ph.fit(train_DAG_Path,ref_node_list,ref_seq,modify_dict,True,windows_length)

def viterbi(train_DAG_Path,Viterbi_DAG_Path,parameter_name,seqiddb,ref_seq,ref_node_list):
    
    windows_length=100
    if not  os.path.exists(Viterbi_DAG_Path+'zip20/'+'graph.pkl'):
        AandZ(Viterbi_DAG_Path,Viterbi_DAG_Path+'zip20/',20)
    if not os.path.exists(Viterbi_DAG_Path+'zip20/'+'v_id.npy'):
        os.system('cp '+Viterbi_DAG_Path+'v_id.npy '+Viterbi_DAG_Path+'zip20/'+'v_id.npy')

    train_times=1
    while True:
        if not os.path.exists(train_DAG_Path+'ini/{}_pc_{}.npy'.format(parameter_name,train_times)):
            break
        parameter_path =train_DAG_Path+'ini/{}_pc_{}.npy'.format(parameter_name,train_times)
        train_times+=1
    ph = Profile_HMM(train_DAG_Path,Viterbi_DAG_Path+'zip20/',parameter_name,parameter_path=parameter_path)
    ph.init_viterbi_data(Viterbi_DAG_Path+'zip20/',ref_node_list,ref_seq,windows_length=windows_length)
    ph.Viterbi(seqiddb)
    ph.state_to_aligment(seqiddb)


def save_ref_node_local(graphPath,thr):
    graph = load_graph(graphPath)
    _ref_seq,_ref_node_list,_emProbMatrix = graph.convertToAliReferenceDAG(thr)
    np.savez(graphPath+'thr_{}.npz'.format(thr),ref_seq=_ref_seq,ref_node_list=_ref_node_list,emProbMatrix=_emProbMatrix)
def save_ref_node_global(graphPath,thr):
    graph = load_graph(graphPath)
    _ref_seq,_ref_node_list,_emProbMatrix = graph.convertToAliReferenceDAG(thr)
    np.savez(graphPath+'thr_{}.npz'.format(thr),ref_seq=_ref_seq,ref_node_list=_ref_node_list,emProbMatrix=_emProbMatrix)
    ori_node_list = np.load(graphPath+'onm.npy',allow_pickle=True)
    ori_node_index = np.load(graphPath+'onm_index.npy',allow_pickle=True)
    ref_onm_list=[]
    for node in _ref_node_list:
        ref_onm_list.append(ori_node_list[ori_node_index[node][0]:ori_node_index[node][1]])
    np.save(graphPath+'ref_onm_list_{}.npy'.format(thr),np.array(ref_onm_list,dtype=object))
    np.save(graphPath+'refseq_{}.npy'.format(thr),_ref_seq)

def find_new_reflist(Viterbi_DAG_Path,ref_nodelist,ref_seq):
    graph = load_graph(Viterbi_DAG_Path,mode='ali')
    ori_reflist=[]
    w_length = graph.fragmentLength
    coor_list = []
    idx=0
    lastcoor=0
    check_flag=0
    for m_node in ref_nodelist:

        seq = ref_seq[idx:idx+w_length]
        idx+=1
        ori_nodes = graph.fragmentNodeDict.get(seq,[])
        refnode = -1
        
        for node in ori_nodes:
            if set(list(graph.nodeList[node][3]))&set(list(m_node))!=set():
                refnode=node
        if refnode!=-1:
            ori_reflist.append(refnode)
            coor_list.append(graph.coordinateList[refnode])
            if graph.coordinateList[refnode]<=lastcoor:
                check_flag=1
            lastcoor = graph.coordinateList[refnode]
            
        else:
            ori_reflist.append(-1)
            coor_list.append(-1)
    
    if check_flag==1:
        Block_list,Block_dif = GRAPH.array_to_block(coor_list)
        cprg = GRAPH.remove_points_to_increase(Block_list)
        for rg in cprg:
            st,ed = rg[0][1], rg[1][1]+1
            for i in range(st,ed):
                ori_reflist[i]=-1
                coor_list[i]=-1
        print(ori_reflist.count(-1),len(ori_reflist),'//!!//',len(ref_seq))

    return ori_reflist
def find_global_ref_node_batch(pathlist,ref_onm_list,ref_seq,thr):
    for outpath in tqdm(pathlist):
        new_reflist = find_new_reflist(outpath,ref_onm_list,ref_seq)
        np.save(outpath+'local_ref_nodelist_{}.npy'.format(thr),new_reflist)
def find_global_ref(subGraphLevel,subGraphNum,train_ref_Path,thr):
    
    ref_onm_list = np.load(train_ref_Path+'ref_onm_list_{}.npy'.format(thr),allow_pickle=True)
    ref_seq = str(np.load(train_ref_Path+'refseq_{}.npy'.format(thr)))
    subGraphPathList=[]
    for graph_name in range(1,subGraphNum+1):
        subGraphPathList.append("/state1/result_DATA_back_up/Merge_Graph/merge_{}/{}/".format(subGraphLevel,graph_name))
    processlist=[]
    pool_num = 50
    for idx in range(pool_num):
        processlist.append(Process(target=find_global_ref_node_batch,args=(subGraphPathList[idx::pool_num],ref_onm_list,ref_seq,thr,)))
    [p.start() for p in processlist]
    [p.join() for p in processlist]



def train_all_data():
    start = datetime.now()
    outpath = "/state1/result_DATA_back_up/Merge_Graph/merge_10/1/"
    newfra=20
    train_DAG_Path=outpath+'zip{}/'.format(newfra)
    # # if not  os.path.exists(train_DAG_Path+'graph.pkl'):
    # AandZ(outpath,train_DAG_Path,newfra)
    # if not os.path.exists(train_DAG_Path+'ini'):
    #     os.makedirs(train_DAG_Path+'ini')
    # if not os.path.exists(train_DAG_Path+'Matrix'):
    #     os.makedirs(train_DAG_Path+'Matrix')
    # for thr in [0.01,0.001]:
    #     save_ref_node_global(outpath,thr)
    #     tarin_ref_path = outpath
    #     find_global_ref(1,425,tarin_ref_path,thr)
    # end = datetime.now()
    # print('Prepare',end-start)
    for pid in range(1,7):
        parameter_name = 'tr{}'.format(pid)
        modify_dict = np.load('/state1/result_DATA_back_up/{}_modifydict.npy'.format(parameter_name),allow_pickle=True).item()
        modify_dict['emProbAdds_Match_head'] = modify_dict['emProbAdds_Match']
        modify_dict['emProbAdds_Match_tail'] = modify_dict['emProbAdds_Match']
        print(modify_dict)
        np.save(train_DAG_Path+'ini/{}_modifydict.npy'.format(parameter_name),modify_dict)
        print()
        train(outpath,train_DAG_Path,parameter_name,modify_dict)
        end = datetime.now()
        print('pid',end-start)
    end = datetime.now()
    print(end-start)

def viterbi_subgraph(index):
    start = datetime.now()
    outpath = "/state1/result_DATA_back_up/Merge_Graph/merge_10/1/"
    newfra=20
    train_DAG_Path=outpath+'zip{}/'.format(newfra)
    subGraphLevel=1

    Viterbi_DAG_Path = "/state1/result_DATA_back_up/Merge_Graph/merge_{}/{}/".format(subGraphLevel,index)

    oriGraphPath = "/state1/result_DATA_back_up/graph_of_Is_complete_230515/"
    oriGraphIdlist = range((index-1)*(2**subGraphLevel)+1,index*(2**subGraphLevel)+1)
    if not os.path.exists(Viterbi_DAG_Path+'osm.db'):
        build_adb_merge(oriGraphPath,Viterbi_DAG_Path,oriGraphIdlist)
    seqiddb = sql_master('',Viterbi_DAG_Path+'osm.db') # ,mode='disk'
    
    for pid in range(1,7):
        parameter_name = 'tr{}'.format(pid)

        modify_dict = np.load('/state1/result_DATA_back_up/{}_modifydict.npy'.format(parameter_name),allow_pickle=True).item()

        modify_dict['emProbAdds_Match_head'] = modify_dict['emProbAdds_Match']
        modify_dict['emProbAdds_Match_tail'] = modify_dict['emProbAdds_Match']

        ref_seq = str(np.load(outpath+'refseq_{}.npy'.format(modify_dict['weight_thr'])))
        ref_node_list = np.load(Viterbi_DAG_Path+'local_ref_nodelist_{}.npy'.format(modify_dict['weight_thr'])).tolist()
        print(modify_dict)
        np.save(train_DAG_Path+'ini/{}_modifydict.npy'.format(parameter_name),modify_dict)
        print()
        viterbi(train_DAG_Path,Viterbi_DAG_Path,parameter_name,seqiddb,ref_seq,ref_node_list)
    end = datetime.now()
    print(end-start)

def zip_all_ali():
    path = '/state1/result_DATA_back_up/Merge_Graph/merge_1/'
    for parameter_name in range(1,7):
        parameter_name='tr{}'.format(parameter_name)
        max_i_length_global = np.load(path+'1/zip20/ali_result_{}/insert_length_dict.npy'.format(parameter_name),allow_pickle=True).item()
        for i in tqdm(range(2,426)):
            # if i !=1 and '.npy' not in i:
            n_max_i_length = np.load(path+'{}/zip20/ali_result_{}/insert_length_dict.npy'.format(i,parameter_name),allow_pickle=True).item()
            max_i_length_global = {key: max(max_i_length_global[key], n_max_i_length[key]) for key in max_i_length_global}
        # print(np.where(np.array(list(max_i_length_global.values()))!=0)[0].size)
        print(path+'insert_length_{}.npy'.format(parameter_name))
        np.save(path+'insert_length_{}.npy'.format(parameter_name),max_i_length_global)

        name_list_global = []
        zipali_global = []
        for i in tqdm(range(1,426)):
            # if '.npy' not in i:
            max_i_length_local = np.load(path+'{}/zip20/ali_result_{}/insert_length_dict.npy'.format(i,parameter_name),allow_pickle=True).item()
            indexdict = np.load(path+'{}/zip20/ali_result_{}/indexdict.npy'.format(i,parameter_name),allow_pickle=True).item()
            zipali = np.load(path+'{}/zip20/ali_result_{}/zipali.npy'.format(i,parameter_name),allow_pickle=True).tolist()
            namelist = np.load(path+'{}/zip20/ali_result_{}/namelist.npy'.format(i,parameter_name),allow_pickle=True).tolist()
            insert_list=[]
            for x in max_i_length_global.keys():
                if x in max_i_length_local.keys():
                    insert_global = max_i_length_global[x]
                    insert_local = max_i_length_local[x]
                    if insert_global>insert_local:
                        # print(x,insert_global-insert_local)
                        # print(indexdict[(2,x)])
                        insert_list.append((indexdict[(2,x)],insert_global-insert_local))
            insert_list.sort(key=lambda x: x[0], reverse=True)
            # print(zipali)
            for j in insert_list:
                for k in range(j[1]):
                    zipali.insert(j[0],[0])
            
            if zipali_global==[]:
                l = len(zipali)
                for j in range(l):
                    ll = len(zipali[j])
                    for k in range(1,ll):
                        zipali[j][k][1] = zipali[j][k][1].tolist()
                zipali_global = zipali
            else:
                add=len(name_list_global)
                # print(add)
                l = len(zipali_global)
                for j in range(l):
                    
                    tmp_dict={}

                    base_lists=zipali[j]
                    for k in range(1,len(base_lists)):
                        base_list = base_lists[k]
                        base = base_list[0]
                        base_array = base_list[1]
                        base_array+=add
                        tmp_dict[base] = tmp_dict.get(base,[])
                        tmp_dict[base]+=base_array.tolist()

                    donebase=set()
                    ll = len(zipali_global[j])
                    for bindex in range(1,ll):
                        base = zipali_global[j][bindex][0]
                        if base in tmp_dict.keys():
                            zipali_global[j][bindex][1] += tmp_dict[base]
                            donebase.add(base)
                    for bindex in tmp_dict.keys()-donebase:
                        zipali_global[j].append([bindex,tmp_dict[bindex]])

            name_list_global.extend(namelist)
        print(len(name_list_global))
        if not os.path.exists(path+parameter_name):
            os.makedirs(path+parameter_name)

        np.save(path+parameter_name+'/zipali.npy',zipali_global)
        # if i==1:

        np.save(path+parameter_name+'/namelist.npy',name_list_global)

def train_and_viterbi_testset(index):
    
    
    fastapath = "/state1/DATA/test_fasta_set_lowN/{}.fasta".format(index)
    outpath = "/state1/result_DATA_back_up/Graph_test2/near_test/{}/".format(index)
    fragmentLength = 64
    build_graph(fastapath,outpath,fragmentLength,maxExtensionLength=5000,graphID=str(index.split('_')[0]),nodeIsolationThreshold=5000)
    newfra=20
    train_DAG_Path=outpath+'zip{}/'.format(newfra)


    start = datetime.now()
    # if not os.path.exists(train_DAG_Path):
    AandZ(outpath,train_DAG_Path,newfra)
    if not os.path.exists(train_DAG_Path+'ini'):
        os.makedirs(train_DAG_Path+'ini')
    if not os.path.exists(train_DAG_Path+'Matrix'):
        os.makedirs(train_DAG_Path+'Matrix')
    if not os.path.exists(outpath+'osm.db'):
        build_adb(outpath,str(index.split('_')[0]))
    seqiddb = sql_master('',outpath+'osm.db') # ,mode='disk'
    for thr in [0.01,0.001]:
        save_ref_node_local(outpath,thr)

    for pid in range(1,7):
        parameter_name = 'tr{}'.format(pid)

        modify_dict = np.load('/state1/result_DATA_back_up/{}_modifydict.npy'.format(parameter_name),allow_pickle=True).item()

        modify_dict['emProbAdds_Match_head'] = modify_dict['emProbAdds_Match']
        modify_dict['emProbAdds_Match_tail'] = modify_dict['emProbAdds_Match']



        np.save(train_DAG_Path+'ini/{}_modifydict.npy'.format(parameter_name),modify_dict)
        print()
        ali(train_DAG_Path,train_DAG_Path,seqiddb,modify_dict,parameter_name,outpath)
    end = datetime.now()
    print(end-start)

if __name__ == '__main__':
    # train_and_viterbi_testset('1_50')
    # exit()
    fastapath = str(sys.argv[1])
    outpath = str(sys.argv[2])
    subgraphID = str(sys.argv[3])
    i_global_random_low = float(sys.argv[4])     # 初始参数，全局变量，随机数下限
    i_global_random_up = float(sys.argv[5])      # 初始参数，全局变量，随机数上限
    m_global_random_low = float(sys.argv[6])     # 更新参数，全局变量，随机数下限
    m_global_random_up = float(sys.argv[7])      # 更新参数，全局变量，随机数上限

    parameter_name='tr1'

    fragmentLength = 64
    build_graph(fastapath,outpath,fragmentLength,maxExtensionLength=5000,graphID=str(subgraphID),nodeIsolationThreshold=5000)
    newfra=20
    train_DAG_Path=outpath+'zip{}/'.format(newfra)


    start = datetime.now()
    if not os.path.exists(train_DAG_Path):
        AandZ(outpath,train_DAG_Path,newfra)
    if not os.path.exists(train_DAG_Path+'ini'):
        os.makedirs(train_DAG_Path+'ini')
    if not os.path.exists(train_DAG_Path+'Matrix'):
        os.makedirs(train_DAG_Path+'Matrix')
    if not os.path.exists(outpath+'osm.db'):
        build_adb(outpath,str(subgraphID))
    seqiddb = sql_master('',outpath+'osm.db')  # ,mode='disk'
    for thr in [0.01,0.001]:
        save_ref_node_local(outpath,thr)
    modify_dict = {}

    modify_dict['emProbAdds_Match'] = -3
    modify_dict['emProbAdds_Match_head'] = modify_dict['emProbAdds_Match']
    modify_dict['emProbAdds_Match_tail'] = modify_dict['emProbAdds_Match']
    modify_dict['init_M2D'] = -4
    modify_dict['init_M2I'] = -4
    modify_dict['init_I2I'] = np.log(1/2)
    modify_dict['init_D2M'] = np.log(1/2)
    modify_dict['init_D2End'] = np.log(1/2)
    modify_dict['weight_thr'] = 0.01
    modify_dict['head_length'] = 50
    modify_dict['tail_length'] = 50
    modify_dict['emProbAdds_Match_head'] = -3
    modify_dict['emProbAdds_Match_tail'] = -3
    modify_dict['trProbAdds_mm'] = -3
    modify_dict['trProbAdds_md'] = -5
    modify_dict['trProbAdds_mi'] = -50
    modify_dict['trProbAdds_PiM'] = -1
    modify_dict['trProbAdds_PiI'] = -1
    modify_dict['trProbAdds_PiD'] = -1
    modify_dict['trProbAdds_im'] = -1
    modify_dict['trProbAdds_ii'] = -1
    modify_dict['trProbAdds_iend'] = -1
    modify_dict['trProbAdds_ii_tail'] = -1
    modify_dict['trProbAdds_mend'] = -10
    modify_dict['trProbAdds_mi_tail'] = -10


    # np.save(train_DAG_Path+'ini/{}_modifydict.npy'.format(parameter_name),modify_dict)
    print()
    ali(train_DAG_Path,train_DAG_Path,seqiddb,modify_dict,parameter_name,outpath)


# 若想norandom，必须注释掉with_random函数，否则在random中直接改会变成inf