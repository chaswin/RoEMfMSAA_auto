
import os
import pickle
from copy import deepcopy
from datetime import datetime
import time
import numpy as np
from alive_progress import alive_bar
from tqdm import tqdm
from multiprocessing import Queue,Pool,Lock,Manager,shared_memory,Process,Value,Array
import sys
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import pandas as pd
import sqlite3
from Bio import SeqIO
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from numba import jit
from collections import deque
from contextlib import contextmanager
import re
from numba.typed import List
from numba.typed import Dict
from numba import types

'''
拓扑序依赖片段图构建程序
图类型子程序
该程序定义图类实例的各种属性的初始值以及各种可调用方法
在定义图时，需要输入一下参数
name:图的名字
fragmentLength:片段的长度
savePath:图保存的路径
seq_id:第一条序列的id
sequence:第一条序列
程序将第一条序列进行初始化程序转化为原始的拓扑序依赖片段图，并创建一个用于查询前后关系的数据库
'''

class GRAPH:
    def __init__(self, inpath,id, fragmentLength,savePath,firstSequenceId,sequence,maxExtensionLength=np.NINF,allBiteofOSM=32,firstBiteofOSM=16,allBiteofONM=64,firstBiteofONM=32):
        # 拓扑序坐标列表
        self.coordinateList = []
        # 节点来源列表
        self.SourceList=np.array([],dtype=object)
        # 后向拓扑序坐标列表
        self.backwardCoordinateList={}
        # 最大延伸值，大于该值的序列会被跳过
        self.maxExtensionLength = maxExtensionLength
        # 延伸长度超过阈值的序列列表
        self.overExtensionSequences=set()
        # 序列路径最终包含图节点的map
        self.sequencePathNodeMap=np.array([],dtype=object)
        # 与图结合度低的序列集合
        self.lowMergeDegreeSequence=set()
        # 图id
        self.graphID = id
        # 初始序列片段长度
        self.originalfragmentLength = int(fragmentLength)
        # 序列片段长度
        self.fragmentLength = int(fragmentLength)
        # 图保存路径
        self.savePath = savePath
        # 片段序列对应节点的字典，用于查找拥有同一片段序列的节点列表
        self.fragmentNodeDict={}
        # 边集合，记录节点的前后关系，在需要查找时先导入至数据库中
        self.edgeWeightDict = dict()
        # 最长路径节点集合
        self.longestPathNodeSet = set()
        # 起始节点的集合
        self.startNodeSet = set()
        # 终止节点的集合
        self.endNodeSet=set()
        # 加入图的序列ID列表
        self.sequenceIDList = []
        # 已经在可查询图中的边的集合
        self.currentEdgeSet = set()
        # 图包含的序列数量
        self.sequenceNum = 0
        # 连接数据库更新信号
        self.queryGraphUpdateFlag =0
        # 图的节点集合
        self.nodeList=[]
        # 总节点数量
        self.totalNodes=0
        # 可查询图的属性（用前向星结构存储为几个列表）
        self.queryGraphHeadList=[]
        self.queryGraphTailList=[]
        self.queryGraphHeadEdges=[]
        self.queryGraphTailEdges=[]

        # 图包含的子图fasta来源
        self.fastaFilePathList = [inpath]

        # 记录用于编码的数据
        self.firstBiteofOSM = firstBiteofOSM
        self.allBiteofOSM = allBiteofOSM
        self.firstBiteofONM = firstBiteofONM
        self.allBiteofONM = allBiteofONM
        self.isolatedSequenceList=set()

        '''
        以下为第一条序列初始化为图的过程
        '''

        w_length = len(sequence)-self.fragmentLength+1
        for row in range(w_length):
            # 切分序列片段
            sequenceFragment =sequence[row:row+self.fragmentLength]
            # 建立节点
            newNode = self.add_new_node(sequenceFragment)
            newNode[0] = row
            newNode[2]+=1
            self.totalNodes+=1
            # 将节点添加到节点列表中
            self.nodeList.append(newNode)
            # 更新可查询图结构
            self.queryGraphHeadList.append(-1)
            self.queryGraphTailList.append(-1)
            # 更新片段-节点字典
            self.fragmentNodeDict.setdefault(newNode[1],[]).append(newNode[0])
            # 坐标列表更新
            self.coordinateList.append(row+1)
            if row > 0:
                self.edgeWeightDict[(row-1, row)]=1

        # 起始终止节点信息更新
        self.nodeList[0][4] += 1
        self.nodeList[0][6]=1
        self.startNodeSet.add(0)
        self.nodeList[row][5] += 1
        self.nodeList[row][6]=1
        self.endNodeSet.add(row)
        
        # 更新最长路径长度
        self.maxlength = row+1

        # 更新序列ID列表和包含完整序列信息
        self.sequenceIDList.append(firstSequenceId)
        self.sequenceNum+=1
        self.sequencePathNodeMap.resize(self.sequencePathNodeMap.size+1,refcheck=True)
        self.sequencePathNodeMap[-1]=np.array(list(range(self.totalNodes)),dtype=np.uint32)
        

    @staticmethod
    def get_first_number(num,firstBite,allBite):
        mask1 = (1 << firstBite) - 1
        # 提取第一个数
        return np.bitwise_and(np.right_shift(num, allBite - firstBite), mask1)
    @staticmethod
    def get_second_number(num,firstBite,allBite):
        mask2 = (1 << (allBite - firstBite)) - 1
        # 提取第二个数
        return np.bitwise_and(num, mask2)
    @staticmethod
    def save_numbers(num1, num2,firstBite,allBite):
        # 存储两个数
        return np.bitwise_or(np.left_shift(num1, allBite - firstBite), num2)
    

    # @staticmethod
    # def get_ori_graph_id(num,n=32):
    #     mask1 = (1 << n) - 1
    #     # 提取第一个数
    #     return np.bitwise_and(np.right_shift(num, 64 - n), mask1)
    
    # @staticmethod
    # def get_ori_node_id(num,n=32):
    #     mask2 = (1 << (64 - n)) - 1
    #     # 提取第二个数
    #     return np.bitwise_and(num, mask2)
    
    # @staticmethod
    # def save_ori_graph_node_id(num1,num2,n=32):
    #     # 存储两个数
    #     return np.bitwise_or(np.left_shift(num1, 64 - n), num2)
    

    @staticmethod
    def array_to_block(arr):
        result = []
        differences = []
        n = len(arr)
        i = 0
        
        while i < n:
            if arr[i] != -1:
                start = arr[i]
                start_index = i
                while i + 1 < n and arr[i + 1] != -1 and arr[i]+1==arr[i+1]:
                    i += 1
                end = arr[i]
                end_index = i
                result.append([[start, start_index], [end, end_index]])
                differences.append(start - start_index)
            i += 1
        
        return result, differences
    
    @staticmethod
    def remove_points_to_increase(lst):
        '''
        动态规划计算锚点取消范围
        '''
        weights=[]
        for i in lst:
            weights.append(i[1][1]-i[0][1]+1)
        n = len(lst)
        dp = [0] * n
        prev = [-1] * n

        for i in range(n):
            dp[i] = weights[i]
            for j in range(i):
                if lst[j][1][0] < lst[i][0][0] and dp[j] + weights[i] > dp[i]:
                    dp[i] = dp[j] + weights[i]
                    prev[i] = j

        max_weight = max(dp)
        index = dp.index(max_weight)
        result = []

        while index != -1:
            result.append(index)
            index = prev[index]

        removed_indices = [i for i in range(n) if i not in result]
        cp_rg=[]
        for i in removed_indices:
            cp_rg.append(lst[i])
        return cp_rg

    @staticmethod
    def Cyclic_Anchor_Combination_Detection(Block_list, Block_dif):
        '''
        潜在环检测
        '''
        cp_rg=[]
        fixflag=0
        block_num = len(Block_dif)
        for i in range(1,block_num):
            if Block_list[i-1][1][0] >= Block_list[i][0][0] :# or Block_dif[i-1]-Block_dif[i]>300
                fixflag=1

        if fixflag==1:
            cp_rg = GRAPH.remove_points_to_increase(Block_list)
        
        return cp_rg

    @staticmethod
    def onm2db(inpath,ori_node_list):
        'free'
        if os.path.exists(inpath+"onm.db"):
            os.remove(inpath+"onm.db")
        
        print(inpath+"onm.db")
        conn = sqlite3.connect(inpath+"onm.db")
        
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS {} (
            node_id INTEGER PRIMARY KEY,
            orinodes BLOB NOT NULL
        )
        '''.format('orinode'))
        st = datetime.now()
        arrays = ori_node_list

        print(len(arrays))
        
        binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]

        df = pd.DataFrame(binary_arrays, columns=['node_id','orinodes'])

        end = datetime.now()
        print('read {}'.format('orinode'),end-st)
        # df.to_sql('vset'+str(i), conn, index=False,dtype={'virus': LargeBinary})
        df.to_sql('orinode', con=conn, if_exists='append', index=False)
        print('orinode')
        print('build db',inpath+"onm.db")

        end = datetime.now()
        print('end {}'.format('orinode'),end-st)
        conn.commit()
        sql = '''CREATE INDEX nodeid ON orinode (node_id);'''
        cursor.execute(sql)
        print('end')
        conn.commit()

        conn.close()

    @staticmethod
    def osm2db(inpath,i='1'):
        'free'
        if os.path.exists(inpath+"osm.db"):
            os.remove(inpath+"osm.db")
        print(inpath+"osm.db")
        conn = sqlite3.connect(inpath+"osm.db")
        
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS {} (
            node_id INTEGER PRIMARY KEY,
            virus BLOB NOT NULL
        )
        '''.format('vset'+str(i)))
        st = datetime.now()
        arrays = np.load(inpath+'osm.npy',allow_pickle=True)

        print(len(arrays))
        
        binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]

        df = pd.DataFrame(binary_arrays, columns=['node_id','virus'])

        end = datetime.now()
        print('read {}'.format(i),end-st)
        df.to_sql('vset'+str(i), con=conn, if_exists='append', index=False)
        print('vset'+str(i))
        print('build db',inpath+"osm.db")

        end = datetime.now()
        print('end {}'.format(i),end-st)
        conn.commit()
        sql = '''CREATE INDEX nodeid{0} ON vset{0} (node_id);'''.format(i)
        cursor.execute(sql)
        print('end')
        conn.commit()

        conn.close()

    
    
    '''
    图结构方法
    '''
    def rebuild_queryGraph(self):
        '''
        从头构建前向星结构
        '''
        totalNodes = self.totalNodes
        edgeWeightDict = self.edgeWeightDict
        queryGraphHeadList = [-1]*totalNodes
        queryGraphTailList = [-1]*totalNodes
        queryGraphHeadEdges=[]
        queryGraphTailEdges=[]
        for idx, i in enumerate(edgeWeightDict.keys()):
            queryGraphHeadEdges.append([i[1], queryGraphHeadList[i[0]],0])
            queryGraphHeadList[i[0]] = idx
            queryGraphTailEdges.append([i[0], queryGraphTailList[i[1]],0])
            queryGraphTailList[i[1]] = idx
        self.queryGraphHeadList = queryGraphHeadList
        self.queryGraphTailList = queryGraphTailList
        self.queryGraphHeadEdges = queryGraphHeadEdges
        self.queryGraphTailEdges = queryGraphTailEdges
        self.queryGraphUpdateFlag=1

    def add_edge_to_queryGraph(self, u, v):
        '''
        用于向前向星结构添加新的边
        '''
        self.queryGraphHeadEdges.append([v, self.queryGraphHeadList[u],0])
        self.queryGraphHeadList[u] = len(self.queryGraphHeadEdges) - 1

        self.queryGraphTailEdges.append([u, self.queryGraphTailList[v],0])
        self.queryGraphTailList[v] = len(self.queryGraphTailEdges) - 1

    def update_queryGraph(self):
        '''
        更新前向星结构，以备查询节点前后关系
        '''
        # 若数据库更新信号为1
        if self.queryGraphUpdateFlag==0:
            if len(self.queryGraphHeadList)!=self.totalNodes:
                self.queryGraphHeadList.extend([-1]*(self.totalNodes-len(self.queryGraphHeadList)))
            if len(self.queryGraphTailList)!=self.totalNodes:
                self.queryGraphTailList.extend([-1]*(self.totalNodes-len(self.queryGraphTailList)))
            # 取得需要更新的连接
            links = self.edgeWeightDict.keys()-self.currentEdgeSet
            # 若更新不为空
            if links!=set():
                
                for link in links:
                    self.add_edge_to_queryGraph(link[0],link[1])
                    
                # 更新已存在于数据库中的连接
                self.currentEdgeSet |= links

            # 前向星信息已更新
            self.queryGraphUpdateFlag=1
    

    
    @staticmethod
    def no_degenerate(sequence):
        return bool(re.fullmatch(r'[ATCG]*', sequence))
    
    @staticmethod
    def is_fragment_matching_criteria(sequence):
        return len(set(sequence))>1 and GRAPH.no_degenerate(sequence)
    
    def sequence_to_path(self,sequence):
        '''
        初始化路径，将序列转化为节点列表
        '''
        # 由序列转化成的节点列表，对应新增的路径
        sequencePathNodeList = []
        # 新路径中节点的坐标列表，在主路径拓扑序检查中仅显示主路径坐标的节点
        anchorCoordinateList = []
        # 图更新信号
        row = 0
        sequenceFragment = sequence[row:row+self.fragmentLength]
        optional_nodes = self.fragmentNodeDict.get(sequenceFragment,[])
        optional_nodes = set(optional_nodes) & self.startNodeSet

        if optional_nodes!=set():
            newNode = self.nodeList[optional_nodes.pop()]
            sequencePathNodeList.append(newNode)
            anchorCoordinateList.append(self.coordinateList[newNode[0]])
        else:
            newNode = self.add_new_node(sequenceFragment)
            sequencePathNodeList.append(newNode)
            anchorCoordinateList.append(-1)


        w_length = len(sequence)-self.fragmentLength

        for row in range(1,w_length):
            # 截取序列片段
            sequenceFragment = sequence[row:row+self.fragmentLength]
            # 若序列特异性较高，则为其选择锚定节点
            if GRAPH.is_fragment_matching_criteria(sequenceFragment):
                # 构建备选节点列表 
                confirmAnchor=False
                alternativeNodes = self.fragmentNodeDict.get(sequenceFragment,[])
                
                if alternativeNodes!=[]:
                    
                    maxWeight = self.nodeList[alternativeNodes[0]][2]
                    same_weights = [self.nodeList[node][2] for node in alternativeNodes].count(maxWeight)
                    optional_node_id = alternativeNodes[0]
                    
                    if self.nodeList[optional_node_id][6]!=1 and same_weights<=1:
                        confirmAnchor=True

                if confirmAnchor==True:
                    newNode = self.nodeList[optional_node_id]
                    # 在路径节点列表中添加该节点
                    sequencePathNodeList.append(newNode)
                    # 在路径坐标列表中添加该节点的坐标
                    anchorCoordinateList.append(self.coordinateList[optional_node_id])
                else:
                    newNode = self.add_new_node(sequenceFragment)
                    sequencePathNodeList.append(newNode)
                    anchorCoordinateList.append(-1)

            # 若碱基种类不多于2，片段序列复杂度太低，无法作为可信任的锚点，因此新建节点，不选择既有节点进行锚定
            else:
                newNode = self.add_new_node(sequenceFragment)
                sequencePathNodeList.append(newNode)
                anchorCoordinateList.append(-1)

        row = len(sequence)-self.fragmentLength
        sequenceFragment = sequence[row:row+self.fragmentLength]
        optional_nodes = self.fragmentNodeDict.get(sequenceFragment,[])
        optional_nodes = set(optional_nodes) & self.endNodeSet

        if optional_nodes!=set():
            newNode = self.nodeList[optional_nodes.pop()]
            sequencePathNodeList.append(newNode)
            anchorCoordinateList.append(self.coordinateList[newNode[0]])
        else:
            newNode = self.add_new_node(sequenceFragment)
            sequencePathNodeList.append(newNode)
            anchorCoordinateList.append(-1)
        return sequencePathNodeList,anchorCoordinateList



    def map_to_OSM(self):
            
        SourceList= np.array([],dtype=object)
        SourceList.resize(self.totalNodes, refcheck=False)
        for node in self.nodeList:
            SourceList[node[0]] = np.zeros(node[2], dtype=np.uint32)
        index_array = np.zeros(self.totalNodes, dtype=np.uint32)
        seq_idx=0
        fragmentLength = self.fragmentLength
        for seq_nodes in tqdm(self.sequencePathNodeMap):
            pathlength = len(seq_nodes)
            seqs_num_array = np.full(pathlength, seq_idx, dtype=np.uint32)
            site_array = np.arange(fragmentLength - 1, fragmentLength - 1 + pathlength, dtype=np.uint32)
            seqid_and_site_array = GRAPH.save_numbers(seqs_num_array, site_array, self.firstBiteofOSM,self.allBiteofOSM)
            for idx, node in enumerate(seq_nodes):
                SourceList[node][index_array[node]] = seqid_and_site_array[idx]
            index_array[seq_nodes] += 1
            seq_idx+=1
        self.SourceList = SourceList

    def join_into_DAG(self,seq_id,sequencePathNodeList,anchorCoordinateList):
        '''
        将准备好的路径加入图中
        '''
        
        sequencePathLength = len(sequencePathNodeList)
        sequencePathCoordinates = [None] * sequencePathLength
        cursor_coor = 0
        coordinateUpdateList = []
        joinConditionFlag = True
        newmax = 0  
        coordinateList = self.coordinateList
        thr = self.maxExtensionLength  
        for index, node in enumerate(sequencePathNodeList):
            if node[0] != 'tmp':
                node_coor = coordinateList[node[0]]
                diff = cursor_coor + 1 - node_coor
                if diff > 0:
                    coordinateUpdateList.append([sequencePathNodeList[index][0], diff,index])
                    if diff > thr:
                        joinConditionFlag = False
                cursor_coor = node_coor
            else:
                cursor_coor += 1
            sequencePathCoordinates[index] = cursor_coor

        if cursor_coor > self.maxlength:
            newmax=cursor_coor
            if diff > thr:
                joinConditionFlag = False

                
        if joinConditionFlag:
            if coordinateUpdateList==[]:
                if sequencePathNodeList[0][0] =='tmp' : 
                    seq = sequencePathNodeList[0][1]
                    nodeids = set(self.fragmentNodeDict.get(seq,[]) )& self.startNodeSet
                    if nodeids:
                        sequencePathNodeList[0] = nodeids.pop()
                
                for i in range(1,sequencePathLength-1):
                    if sequencePathNodeList[i][0] =='tmp' :
                        seq = sequencePathNodeList[i][1]
                        nodeids = self.fragmentNodeDict.get(seq,[])
                        nid = [self.nodeList[nodeid] for nodeid in nodeids if coordinateList[nodeid] == sequencePathCoordinates[i] and self.nodeList[nodeid][6]!=1]
                        if nid:
                            sequencePathNodeList[i] = nid[0]

                if sequencePathNodeList[-1][0] =='tmp' : 
                    seq = sequencePathNodeList[-1][1]
                    nodeids = set(self.fragmentNodeDict.get(seq,[])) & self.endNodeSet
                    nid = [self.nodeList[nodeid] for nodeid in nodeids if coordinateList[nodeid] == sequencePathCoordinates[-1]]
                    if nid:
                        sequencePathNodeList[-1] = nid[0]


            newnodes=[]
            for index,node in enumerate(sequencePathNodeList):

                if node[0]=='tmp':
                    node[0] = self.totalNodes
                    self.fragmentNodeDict.setdefault(node[1],[]).append(node[0])
                    newnodes.append(node)
                    self.coordinateList.append(sequencePathCoordinates[index])
                    self.totalNodes+=1
                    node[2]+=1
                else:
                    node[2]+=1
                    if self.nodeList[self.fragmentNodeDict[node[1]][0]][2]<node[2]:
                        self.fragmentNodeDict[node[1]].remove(node[0])
                        self.fragmentNodeDict[node[1]].insert(0,node[0])
            self.nodeList.extend(newnodes)

            # 节点包含的首端数+1
            sequencePathNodeList[0][4] += 1
            sequencePathNodeList[0][6]=1

            # 遍历节点
            for i in range(1,sequencePathLength):

                newlink = (sequencePathNodeList[i-1][0], sequencePathNodeList[i][0])
                self.edgeWeightDict[newlink] = self.edgeWeightDict.get(newlink, 0) + 1
            sequencePathNodeList[i][6]=1
            # 节点包含的尾端数+1
            sequencePathNodeList[i][5] += 1

            self.startNodeSet.add(sequencePathNodeList[0][0])
            self.endNodeSet.add(sequencePathNodeList[-1][0])



            self.sequenceIDList.append(seq_id)

            # 若存在检查信号，更新确认图坐标

            if coordinateUpdateList!=[]:
                self.update_queryGraph()
                maxupdate = self.updateCoordiante(update_stem=False,error_nodes=[ex[0] for ex in coordinateUpdateList])
                if maxupdate:
                    print('Extend ',coordinateUpdateList,self.sequenceNum)
                    print('newmaxlength:',self.maxlength)
                    print('---------------------------------------------')

            # 数据库更新信号重置
            self.queryGraphUpdateFlag = 0
            if newmax!=0:
                self.maxlength = newmax
                print('Tail Extend')
                print('newmaxlength:',self.maxlength)
                print('---------------------------------------------')

            self.sequencePathNodeMap.resize(self.sequencePathNodeMap.size+1,refcheck=True)
            self.sequencePathNodeMap[-1]=np.array([n[0] for n in sequencePathNodeList],dtype=np.uint32)
            self.sequenceNum += 1

        else:

            print(GRAPH.array_to_block(anchorCoordinateList))
            print('special sequence',seq_id)
            # 被跳过序列集合中添加该序列的id
            print(coordinateUpdateList)
            self.overExtensionSequences.add(seq_id)
            print('%'*45)

    

    def add_seq(self,seq,seq_id):
        '''
        该子程序用于向图中添加序列
        '''
        sequencePathNodeList,anchorCoordinateList = self.sequence_to_path(seq)
        # 主路径段拓扑序维护
        sequencePathNodeList,anchorCoordinateList = self.Repair_topology(sequencePathNodeList,anchorCoordinateList)

        self.join_into_DAG(seq_id,sequencePathNodeList,anchorCoordinateList)

    def merge_node(self,nodeA,nodeB,mode='build'):
        
        '''
        合并节点，转移合并节点信息
        '''
        self.nodeList[nodeA][4] += self.nodeList[nodeB][4]
        self.nodeList[nodeA][5] += self.nodeList[nodeB][5]
        self.nodeList[nodeA][2] += self.nodeList[nodeB][2]
        if mode!= 'build':
            self.nodeList[nodeA][3] += self.nodeList[nodeB][3]
        else:
            oldsize = self.SourceList[nodeA].size
            self.SourceList[nodeA].resize(oldsize+self.SourceList[nodeB].size)
            self.SourceList[nodeA][oldsize:] = self.SourceList[nodeB]
            self.SourceList[nodeB]=0

    def merge_node_batch(self,merge_list,mode='build'):
        '''
        批量合并节点
        ''' 
        delList=[]
        nodeList = self.nodeList
        for tup in tqdm(merge_list):
            nodeA = tup.pop()
            delList.extend(tup)
            fra_list = []
            size_list = []
            for nodeB in tup:
                nodeList[nodeA][4] += nodeList[nodeB][4]
                nodeList[nodeA][5] += nodeList[nodeB][5]
                nodeList[nodeA][2] += nodeList[nodeB][2]
                if mode!='build':
                    nodeList[nodeA][3] += nodeList[nodeB][3]
                else:
                    size_list.append(self.SourceList[nodeB].size)
                    fra_list.append(self.SourceList[nodeB])
                    self.SourceList[nodeB]=0
            if mode=='build':
                oldsize = self.SourceList[nodeA].size
                self.SourceList[nodeA].resize(oldsize+np.sum(size_list))
                cursor = oldsize
                for idx,fras in enumerate(fra_list):
                    self.SourceList[nodeA][cursor:cursor+size_list[idx]] = fras
                    cursor+=size_list[idx]
        self.nodeList = nodeList
        return delList



    def rerange_node(self,mode='build',commonBaseThreshold=-1,commonBaseTypeThreshold=2):
        '''
        查找坐标相同的同序列节点，将其合并
        '''
        if commonBaseThreshold==-1:
            commonBaseThreshold = self.fragmentLength
        print(commonBaseThreshold,'fragmentLength',self.fragmentLength)

        merge_list=Manager().list()
        lock=Lock()
        change_dict_share = Array('i', range(self.totalNodes))
        replaceDict = np.frombuffer(change_dict_share.get_obj(), dtype=np.int32)
        replaceDict = replaceDict.reshape(self.totalNodes)
        def check_merge_list(subklist):
            fragmentNodeDict = self.fragmentNodeDict
            DAG_nodeList = self.nodeList
            coordinateList = self.coordinateList
            tmp_merge_list=[]
            nodeList = [fragmentNodeDict[i] for i in subklist  if  len(fragmentNodeDict[i])>1 and len(set(i)&set(['A','T','C','G']))>=commonBaseTypeThreshold and sum([i.count(aa) for aa in "ATCG"])>=commonBaseThreshold]
            for i in nodeList:
                tlist={}
                for j in [j for j in i if DAG_nodeList[j][6]!=1]:
                    tlist.setdefault(coordinateList[j],[]).append(j)
                for tup in tlist.values():
                    
                    if len(tup)>1:
                        with lock:
                            replaceDict[np.array(tup[:-1])]=tup[-1]

                        tmp_merge_list.append(tup)
            with lock:
                merge_list.extend(tmp_merge_list)
        processlist=[]
        fragmentList = list(self.fragmentNodeDict.keys())
        pool_num = 10
        for idx in range(pool_num):
            processlist.append(Process(target=check_merge_list,args=(fragmentList[idx::pool_num], )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]


        if len(merge_list)!=0:
            newlinkset = {}
            for i in tqdm(self.edgeWeightDict.keys()):
                k = (int(replaceDict[i[0]]),int(replaceDict[i[1]]))
                newlinkset[k] = newlinkset.get(k,0)+self.edgeWeightDict[i]
            self.edgeWeightDict = newlinkset

        
        delList = self.merge_node_batch(merge_list,mode=mode)

        self.fast_remove(delList)

        if delList!=[]:
            self.reorderNodelist(mode)

        return len(delList)
    
    def rerange_node_back(self,mode='merge',commonBaseThreshold=-1,commonBaseTypeThreshold=2):
        '''
        查找坐标相同的同序列节点，将其合并
        '''

        if commonBaseThreshold == -1:
            commonBaseThreshold = self.fragmentLength
        
        merge_list=Manager().list()
        lock=Lock()
        change_dict_share = Array('i', range(self.totalNodes))
        replaceDict = np.frombuffer(change_dict_share.get_obj(), dtype=np.int32)
        replaceDict = replaceDict.reshape(self.totalNodes)

        def check_merge_list(subkilst):
            fragmentNodeDict = self.fragmentNodeDict
            DAG_nodeList = self.nodeList
            backwardCoordinateList = self.backwardCoordinateList
            tmp_merge_list=[]
            nodeList  = [fragmentNodeDict[i] for i in subkilst  if  len(fragmentNodeDict[i])>1 and len(set(i)&set(['A','T','C','G']))>=commonBaseTypeThreshold and sum([i.count(aa) for aa in "ATCG"])>=commonBaseThreshold]
            for i in nodeList:
                tlist={}
                for j in [j for j in i if DAG_nodeList[j][6]!=1]:
                    tlist.setdefault(backwardCoordinateList[j],[]).append(j)

                for tup in tlist.values():
                    if len(tup)>1:
                        with lock:
                            replaceDict[np.array(tup[:-1])]=tup[-1]
                        tmp_merge_list.append(tup)
            with lock:
                merge_list.extend(tmp_merge_list)
        processlist=[]
        fragmentList = list(self.fragmentNodeDict.keys())
        pool_num = 10
        for idx in range(pool_num):
            processlist.append(Process(target=check_merge_list,args=(fragmentList[idx::pool_num], )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]

        if len(merge_list)!=0:
            newlinkset = {}
            for i in tqdm(self.edgeWeightDict.keys()):
                k = (int(replaceDict[i[0]]),int(replaceDict[i[1]]))
                newlinkset[k] = newlinkset.get(k,0)+self.edgeWeightDict[i]
            self.edgeWeightDict = newlinkset


        delList = self.merge_node_batch(merge_list,mode=mode)

        self.fast_remove(delList)

        if delList!=[]:
            self.reorderNodelist(mode)

        return len(delList)


  
    def removeDegenerateBasePaths(self):
        '''
        删除含有简并碱基的路径
        '''
        bannedSet=set()
        startNode = set()
        endNode = set()

        for node in self.nodeList:
            if node[4]>0:
                startNode.add(node[0])
            elif node[5]>0:
                endNode.add(node[0])

            if node[1][-1] not in {'A','T','C','G'}:
                bannedSet.add(node[0])
                startNode-={node[0]}
                endNode-={node[0]}

        setA = self.findDescendantNodes(list(startNode),bannedSet)
        setB = self.findAncestorNodes(list(endNode),bannedSet)
        allset = setA&setB
        delset = set(range(self.totalNodes))-allset
        if allset==set():
            return 0
        print('start kill degenerate base',len(allset))
        self.edgeWeightDict = {link:value for link,value in self.edgeWeightDict.items() if not set(link) & delset}

        self.fast_remove(delset)

        self.id_mapping =self.reorderNodelist()
        return 1


    
    
    def snapshot(self,savePath=''):
        'free'
        '''
        保存片段图中间态
        '''
        if savePath=='':
            temp_savepath = self.savePath+str(self.sequenceNum)+'/'
        else:
            temp_savepath = savePath
        old_savepath = deepcopy(self.savePath)
        
        if not os.path.exists(temp_savepath):
            os.makedirs(temp_savepath)
        self.savePath = temp_savepath       
        graphfile = open(temp_savepath+'graph.pkl', 'wb')
        pickle.dump(self, graphfile)
        self.savePath = old_savepath
        graphfile.close()
        
    def merge_check(self,mode='build',coordDistanceLimit=50,commonBaseThreshold=-1):
        
        '''
        融合无环路异常的同序列片段节点
        '''
        if commonBaseThreshold==-1:
            commonBaseThreshold = self.fragmentLength
        delList=[]
        print()
        print('same_segment_nodes merge checking...')
        self.coordinateList = np.array(self.coordinateList)
        tupdict={}
        for i in [i for seq,i in self.fragmentNodeDict.items() if len(i)>1 and len(set(seq)&set(['A','T','C','G']))>1 and sum([seq.count(aa) for aa in "ATCG"])>=commonBaseThreshold]: 
            i=list(i)
            tlist=[]
            i_num = len(i)
            for j in range(i_num):
                for k in range(j+1,i_num):
                    tlist.append((i[j],i[k],abs(self.coordinateList[i[j]]-self.coordinateList[i[k]])))

            tlist = sorted(tlist, key=lambda node: node[2], reverse=False)
            if tlist!=[]:
                tupdict.setdefault(tlist[0][2],[]).append([tlist,i])

        tupdict = sorted(tupdict.items(), key=lambda node: node[0], reverse=False)

        
        merge_node_num=0
        
        for seq_tups in tqdm(tupdict):
            
            seq_tup = seq_tups[1]
            for tlist_and_i in seq_tup:
                chooselist=deepcopy(tlist_and_i[1])
                tlist=tlist_and_i[0]
                for tup in tlist:
                    
                    j = tup[0]
                    k = tup[1]
                    if j in chooselist and k in chooselist:

                        if self.coordinateList[j] == self.coordinateList[k]:
                            
                            st1,st2 = self.nodeList[j][4],self.nodeList[k][4] 
                            ed1,ed2 = self.nodeList[j][5],self.nodeList[k][5]
                            if not ((st1 > 0 and st2 > 0) or (st1 <= 0 and st2 <= 0)) and ((ed1 > 0 and ed2 > 0) or (ed1 <= 0 and ed2 <= 0)):
                                continue
                            
                            if k not in chooselist:
                                Cnode =j
                                j =k
                                k=Cnode

                            chooselist.remove(k)

                            self.merge_node_in_queryGraph(j,k)
                            self.merge_node(j,k,mode=mode)
                            delList.append(k)
                            merge_node_num+=1
                            

                        else:
                            
                            if abs(self.coordinateList[j]-self.coordinateList[k])<coordDistanceLimit:
                                
                                if self.coordinateList[j] > self.coordinateList[k]:
                                    Anode = j
                                    Bnode = k
                                else:
                                    Anode = k
                                    Bnode = j
                                endindex = self.coordinateList[Anode]
                                
                                st1,st2 = self.nodeList[j][4],self.nodeList[k][4] 
                                ed1,ed2 = self.nodeList[j][5],self.nodeList[k][5]
                                if not ((st1 > 0 and st2 > 0) or (st1 <= 0 and st2 <= 0)) and ((ed1 > 0 and ed2 > 0) or (ed1 <= 0 and ed2 <= 0)):
                                    continue

                                if self.nonCyclic(Anode,endindex,Bnode):

                                    if Bnode  not in chooselist:
                                        Cnode =''
                                        Cnode =Anode
                                        Anode =Bnode
                                        Bnode=Cnode
                                    chooselist.remove(Bnode)
                                    self.merge_node_in_queryGraph(Anode,Bnode)
                                    
                                    
                                    
                                    adds = max(self.coordinateList[Anode],self.coordinateList[Bnode])-min(self.coordinateList[Anode],self.coordinateList[Bnode])
                                    
                                    if adds==1:
                                        self.coordinateList[Anode] = min(self.coordinateList[Anode],self.coordinateList[Bnode])
                                        pushnodes = list(self.findConsecutiveDescendantNodes([Anode]))
                                        self.coordinateList[pushnodes] += adds
                                    else:
                                        self.coordinateList[Anode] = max(self.coordinateList[Anode],self.coordinateList[Bnode])
                                        stem_push=0
                                        index = self.coordinateList[Anode]+1
                                        now = [node for node in self.findChildNodes([Anode]) if self.coordinateList[node] <index]
                                        while len(now)!=0:
                                            if len(now)>20000:
                                                self.coordinateList[now] = index
                                                stem_push=1
                                                print(index,len(now),adds,'!!!!!!!!!!!!!!!!')
                                                break
                                            self.coordinateList[now] = index
                                            index+=1
                                            now =  ([node for node in self.findChildNodes(now) if self.coordinateList[node] <index])
                                            

                                        if stem_push==1:
                                        # self.coordinateList[Anode] = max(self.coordinateList[Anode],self.coordinateList[Bnode])
                                        # index = self.coordinateList[Anode]+1
                                        # now = [node for node in self.findChildNodes([Anode]) if self.coordinateList[node] <index]
                                            self.local_update_coordinate(now)

                                    self.merge_node(Anode,Bnode,mode=mode)
                                    
                                    delList.append(Bnode)
                                    merge_node_num+=1
                                
        self.coordinateList = list(self.coordinateList)
        if merge_node_num==0:
            return False
        self.fast_remove(delList)
        self.reorderNodelist(mode)
        self.updateCoordiante(mode=mode)
        return True
        


    def reorderNodelist(self,mode='notbuild'):
        '''
        当图中节点有被删除的情况，重置图
        '''
        new_nodelist = []
        id_mapping = {}
        
        new_coordinateList=[]
        self.fragmentNodeDict={}
        index=0
        if mode=='build':
            newSourceList = np.array([],dtype=object)
            newSourceList.resize(self.totalNodes,refcheck=False)
            for node in tqdm(self.nodeList):
                
                if node:
                    id_mapping[node[0]] = index

                    new_coordinateList.append(self.coordinateList[node[0]])
                    
                    newSourceList[index] = self.SourceList[node[0]].copy()
                    node[0] = index
                    new_nodelist.append(node)
                    self.fragmentNodeDict.setdefault(node[1],[]).append(node[0])
                    index+=1
            self.SourceList = newSourceList[:index].copy()
        else:
            for node in self.nodeList:
                if node:
                    nodeid=node[0]
                    id_mapping[nodeid] = index
                    new_coordinateList.append(self.coordinateList[nodeid])              
                    node[0] = index
                    new_nodelist.append(node)
                    self.fragmentNodeDict.setdefault(node[1],[]).append(node[0])
                    index+=1

        self.coordinateList = new_coordinateList 
        self.nodeList=new_nodelist 
        self.edgeWeightDict = { (id_mapping[link[0]], id_mapping[link[1]]): value for (link, value) in self.edgeWeightDict.items()}
        self.startNodeSet = set([id_mapping[node] for node in self.startNodeSet]) 
        self.endNodeSet = set([id_mapping[node] for node in self.endNodeSet]) 
        self.totalNodes = len(self.nodeList)
        self.rebuild_queryGraph()

        if mode=='build':
            self.calculateCoordinates()

        return id_mapping
    
    def reorder_ref_graph_nodes(self,node_set):
        '''
        当图中节点有被删除的情况，重置图
        '''
        new_nodelist = []
        id_mapping = {}
        
        new_coordinateList=[]
        self.fragmentNodeDict={}
        index=0
        
        for node in node_set:
            node = self.nodeList[node]
            nodeid=node[0]
            id_mapping[nodeid] = index
            new_coordinateList.append(self.coordinateList[nodeid])              
            node[0] = index
            new_nodelist.append(node)
            self.fragmentNodeDict.setdefault(node[1],[]).append(node[0])
            index+=1
        self.coordinateList = new_coordinateList 
        self.nodeList=new_nodelist 

        self.edgeWeightDict = { (id_mapping[link[0]], id_mapping[link[1]]): value for (link, value) in self.edgeWeightDict.items()} 

        self.longestPathNodeSet = set() 
        self.startNodeSet = set() 
        self.endNodeSet=set()
        self.totalNodes = len(self.nodeList)
        self.rebuild_queryGraph()
        for node in range(self.totalNodes):
            if self.findParentNodes(node)==[]:
                self.startNodeSet.add(node)
            elif self.findChildNodes(node)==[]:
                self.endNodeSet.add(node)
        self.calculateCoordinates()
        self.findLongestPath()

        return id_mapping
    
    
    def remove_free_seq(self,thr):

        '''
        删除和图连接松散的路径
        '''
        
        reflist = self.findMainPathNodes()
        self.calculateStateRange(reflist,mode='build')
        specialSequenceIndex = set()
        isolatedSequenceIndex=set()
        for nodeid in range(self.totalNodes):

            if self.ref_coor[nodeid]==[0,len(reflist)]:
                specialSequenceIndex|=set([GRAPH.get_first_number(sequenceIDList,self.firstBiteofOSM,self.allBiteofOSM) for sequenceIDList in self.SourceList[nodeid]])
                isolatedSequenceIndex|=set([GRAPH.get_first_number(sequenceIDList,self.firstBiteofOSM,self.allBiteofOSM) for sequenceIDList in self.SourceList[nodeid]])
            elif (self.ref_coor[nodeid][1]-self.ref_coor[nodeid][0])>thr:
                # print(nodeid)
                specialSequenceIndex|=set([GRAPH.get_first_number(sequenceIDList,self.firstBiteofOSM,self.allBiteofOSM) for sequenceIDList in self.SourceList[nodeid]])

        now = set()
        for nodeid in self.startNodeSet:
            if set(GRAPH.get_first_number(self.SourceList[nodeid],self.firstBiteofOSM,self.allBiteofOSM)) & specialSequenceIndex !=set():
                now.add(nodeid)

        specialSequenceIndex = list(specialSequenceIndex)
        print('The number of outlier sequences: ',len(isolatedSequenceIndex))
        print('The number of weakly associated sequences: ',len(specialSequenceIndex))

        if len(specialSequenceIndex)>0:

            delnodelist=[]
            delList =np.array(specialSequenceIndex)
            print(delList)
            for idx in delList:
                for node in self.sequencePathNodeMap[idx]:

                    self.nodeList[node][2]-=1
                    if self.nodeList[node][2]==0:
                        delnodelist.append(node)
            self.sequencePathNodeMap = np.delete(self.sequencePathNodeMap, delList, axis=0)
            self.map_to_OSM()
            self.fast_remove(delnodelist)
            self.edgeWeightDict = {link:value for link,value in self.edgeWeightDict.items() if not set(link) & set(delnodelist)}
            self.reorderNodelist(mode='build')
            for idx in  set(specialSequenceIndex)-isolatedSequenceIndex:
                self.lowMergeDegreeSequence.add(self.sequenceIDList[idx])
            for idx in isolatedSequenceIndex:
                self.isolatedSequenceList.add(self.sequenceIDList[idx])

            self.sequenceIDList = [item for idx, item in enumerate(self.sequenceIDList) if idx not in delList]
            self.sequenceNum-=len(specialSequenceIndex)
        return []
        
    def save_graph(self,savepath='',mode = 'build'):
        '''
        保存片段图
        '''
        if savepath!='':
            if not os.path.exists(savepath):
                os.makedirs(savepath) 
            self.savePath = savepath
        else:
            savepath = self.savePath

        if mode=='build':
            self.merge_check(mode=mode)
            self.mergeNodesWithSameCoordinates(self.fragmentLength,mode=mode,commonBaseThreshold=0,commonBaseTypeThreshold=0)
            
            ori_node_list = np.full((self.totalNodes),0)
            onmindex = np.full((self.totalNodes,2),0)                                                 
            for node in self.nodeList:
                ori_node_list[node[0]] = GRAPH.save_numbers(int(self.graphID),node[0],self.firstBiteofONM,self.allBiteofONM)
                onmindex[node[0]] = np.array([node[0],node[0]+1])
            ori_node_list = np.array(ori_node_list,dtype=object)
            
            
            self.reflist = self.findMainPathNodes()
            self.calculateStateRange(self.reflist,mode='build')

            state_range_dict = {}
            for node in range(self.totalNodes):
                state_range_dict.setdefault(self.ref_coor[node][1]-self.ref_coor[node][0],[]).append(node)
            
            new_vid=[]
            for idx,v in enumerate(self.sequenceIDList):
                key = f"{self.graphID}_{idx}"
                new_vid.append([v,key])

            np.save(self.savePath+'osm.npy',self.SourceList)
            np.save(self.savePath+'onm.npy',ori_node_list)
            np.save(self.savePath+'onm_index.npy',onmindex)
            np.save(self.savePath+'v_id.npy',new_vid)
            print()
            print('//////////////Basic information of graphs//////////////////')
            print('The longest path length in the graph : ',self.maxlength)
            print('The number of nodes in the graph : ',self.totalNodes)
            print('The number of links in the graph: ',len(self.edgeWeightDict))
            print('The number of stem_nodes in the graph: ',len(self.longestPathNodeSet))
            print('ref_sequence_length: ',len(self.reflist)+self.fragmentLength-1)
            print('max_range of nodes in graph:',max(list(state_range_dict.keys())))
            print('graph file save in ',self.savePath)
        else:
            

            ori_node_list = []                                                        
            for node in self.nodeList:
                ori_node_list.append(node[3])
                if len(set(node[3]))!=len(node[3]):
                    print(node[0],'//')
                node[3] = ''

            ori_node_list = np.array(ori_node_list,dtype=object)
            np.save(savepath+'Traceability_path.npy',ori_node_list)

        self.nodeList = np.array(self.nodeList,dtype=object)
        self.coordinateList = np.array(self.coordinateList,dtype=np.uint16)
        self.edgeWeightDict = np.array([[link[0], link[1], value] for (link, value) in self.edgeWeightDict.items()],dtype=object) # self.edgeWeightDict √ list
        self.longestPathNodeSet = np.array(list(self.longestPathNodeSet),dtype=np.uint32) # self.longestPathNodeSet √ array
        self.startNodeSet = np.array(list(self.startNodeSet),dtype=np.uint32) # self.startNodeSet √ array
        self.endNodeSet = np.array(list(self.endNodeSet),dtype=np.uint32) # self.endNodeSet √ array
        # 保存节点列表，坐标列表，边权重字典，最长路径节点，起始路径节点，终止路径节点等信息
        np.savez(savepath+'data.npz', edgeWeightDict=self.edgeWeightDict,coordinateList = self.coordinateList,nodeList = self.nodeList,longestPathNodeSet=self.longestPathNodeSet,startNodeSet=self.startNodeSet,endNodeSet=self.endNodeSet)
        
        print('saving')

        all_attributes = self.__dict__.keys()
        # 定义一个白名单，包含想要保留的属性

        whitelist = ["overExtensionSequences", "lowMergeDegreeSequence","graphID","originalfragmentLength","fragmentLength","savePath","sequenceNum","fastaFilePathList","firstBiteofOSM","allBiteofOSM","firstBiteofONM","allBiteofONM","isolatedSequenceList","totalNodes"] 
        for attr in list(all_attributes):
            if attr not in whitelist:
                delattr(self, attr)

        with open(savepath+'graph.pkl', 'wb') as graphfile: #保存图（包括图ID，原始片段长度，片段长度，保存路径，序列数量，来源fasta文件路径，OSM第一位字节数，OSM总字节数，ONM第一位字节数，ONM总字节数,总节点数，被跳过序列ID列表等信息。
            pickle.dump(self, graphfile)

        print('saved')
        

    def Cyclic_Anchor_Combination_Exclusion(self,copyrglist,sequencePathNodeList,anchorCoordinateList):
        '''
        该子程序用于通过替换或新建手段修复拓扑序异常
        '''
        # 自左向右遍历调整范围块列表内的所有块
        for i in copyrglist:
            start, end = i[0][1], i[1][1]+1
            sequencePathNodeList[start:end] = [self.add_new_node(sequencePathNodeList[j][1]) if not anchorCoordinateList[j]==-1 else sequencePathNodeList[j] for j in range(start, end)]
            anchorCoordinateList[start:end] = [-1 if not anchorCoordinateList[j]==-1 else anchorCoordinateList[j] for j in range(start, end)]

    
    def Repair_topology(self,sequencePathNodeList,anchorCoordinateList):
        '''
        主路径拓扑序维护
        坐标列表建立 >> 确定修复范围 >> 拓扑异常修复
        '''
        
        Coordinate_block_list, Coordinate_block_dif = GRAPH.array_to_block(anchorCoordinateList)
        copyrglist = GRAPH.Cyclic_Anchor_Combination_Detection(Coordinate_block_list, Coordinate_block_dif)

        while copyrglist!=[]:
            # 对锚定调整范围内的节点依次根据调整坐标下限和上限进行调整
            self.Cyclic_Anchor_Combination_Exclusion(copyrglist,sequencePathNodeList,anchorCoordinateList)
            # 根据路径坐标列表生成坐标列表块，以及每个坐标列表块对应的图-路径坐标差
            Coordinate_block_list, Coordinate_block_dif = GRAPH.array_to_block(anchorCoordinateList)
            copyrglist = GRAPH.Cyclic_Anchor_Combination_Detection(Coordinate_block_list, Coordinate_block_dif)
        return sequencePathNodeList,anchorCoordinateList
    
    def add_new_node(self,seq):

        '''
        在图中建立新节点
        '''
        node = ['tmp',seq,0,[],0,0,0]        

        return node
    

    def fast_remove(self,delList):
        for node in delList:
            self.nodeList[node]=[]
        self.startNodeSet -= set(delList)
        self.endNodeSet -= set(delList)
    

    

    def Possible_paths(self):
        'free'
        '''
        计算所有可能路径数
        '''
        self.path_num_list = [0]*self.totalNodes
        for i in self.startNodeSet:
            self.path_num_list[i]=1
        self.forward(self.get_possible_path_forward,self.startNodeSet)
        
        all_path=0
        for fnode in self.endNodeSet:
            all_path+=self.path_num_list[fnode]
        del self.path_num_list
        return all_path

    def mergeNodesWithSameCoordinates(self,fragmentLength,mode='merge',commonBaseThreshold=-1,commonBaseTypeThreshold=2):
        print('same coordinate and segment nodes merge checking...')
        if self.fragmentLength>fragmentLength:
            self.fragmentReduce(fragmentLength)

        merge_nodes_f = self.rerange_node(mode,commonBaseThreshold=commonBaseThreshold,commonBaseTypeThreshold=commonBaseTypeThreshold)
        self.calculateCoordinates_backward()

        merge_nodes_b=self.rerange_node_back(mode,commonBaseThreshold=commonBaseThreshold,commonBaseTypeThreshold=commonBaseTypeThreshold)
        if merge_nodes_b!=0:
            self.calculateCoordinates()
        merge_nodes = merge_nodes_f+merge_nodes_b
        print('The number of nodes in the graph after zip: ',len(self.nodeList))
        if merge_nodes!=0:
            self.mergeNodesWithSameCoordinates(fragmentLength,mode=mode,commonBaseThreshold=commonBaseThreshold,commonBaseTypeThreshold=commonBaseTypeThreshold)
    
    
    def updateCoordiante(self,update_stem=True,error_nodes=[],mode='build'):
        '''
        查找图中坐标拓扑序的连接并尝试修复坐标
        '''
        linksource=self.edgeWeightDict.keys()
        if error_nodes==[]:
            error_nodes = [i[1] for i in linksource if self.coordinateList[i[0]] >= self.coordinateList[i[1]]]
        self.local_update_coordinate(error_nodes)
        maxlength_update=False
        if self.maxlength != max(self.coordinateList):
            self.maxlength = max(self.coordinateList)
            maxlength_update=True
        if update_stem==True:
            self.findLongestPath()
        return maxlength_update

    def findLongestPath(self):
        '''
        该子程序用于更新拓扑主路径
        '''
        
        indexdict = {}
        for node in self.endNodeSet:
            indexdict.setdefault(self.coordinateList[node],[]).append(node)
        
        # 确认最大坐标值
        maxindex = max(indexdict.keys())
        self.maxlength = maxindex
        # 确认主路径头尾节点
        mainend = indexdict[maxindex]

        # 查找主路径尾节点所有连续的父节点
        self.longestPathNodeSet = self.findConsecutiveAncestorNodes(mainend)

    def fragmentReduce(self,newLength=1):
        """
        序列片段缩减
        """
        
        if self.fragmentLength>newLength:
            self.fragmentLength = newLength
            nodes = range(self.totalNodes)
            for node_id in nodes:
                node = self.nodeList[node_id]
                
                if node[0] in self.startNodeSet:
                    node[6]=1
                    sequence = node[1]
                    node[1] = sequence[-newLength:]
                    self.coordinateList[node[0]] +=len(sequence)-newLength
                    nextnode = node[0]
                    w_length =len(sequence)-newLength+1
                    for i in range(1,w_length):
                        seq = sequence[-newLength-i:-i]
                        newNode = self.add_new_node(seq)
                        newNode[0] =self.totalNodes
                        self.totalNodes+=1
                        self.nodeList.append(newNode)
                        self.queryGraphHeadList.append(-1)
                        self.queryGraphTailList.append(-1)
                        newNode[6] = 1
                        newNode[3] = node[3]
                        self.coordinateList.append(self.coordinateList[node[0]]-i)
                        newNode[2] = node[2]
                        self.edgeWeightDict[(newNode[0],nextnode)]=node[2]
                        self.add_edge_to_queryGraph(newNode[0],nextnode)
                        nextnode = newNode[0]
                    newNode[4]=node[4]
                    node[4] = 0
                    self.startNodeSet.remove(node[0])
                    self.startNodeSet.add(newNode[0])

                else:
                    sequence = node[1]
                    node[1] = sequence[-newLength:]
                    self.coordinateList[node[0]] +=len(sequence)-newLength
            self.fragmentNodeDict={}
            for node in self.nodeList:
                self.fragmentNodeDict.setdefault(node[1],[]).append(node[0])


    
    def merge_node_in_queryGraph(self,j,k):
        
        fathernodes = self.findParentNodes(k)
        sonnodes = self.findChildNodes(k)

        for child in sonnodes:
            self.edgeWeightDict[(j, child)] = self.edgeWeightDict.get((j, child), 0) + self.edgeWeightDict[(k, child)]
            del self.edgeWeightDict[(k, child)]
        for parent in fathernodes:
            self.edgeWeightDict[(parent,j)] = self.edgeWeightDict.get((parent,j),0) + self.edgeWeightDict[(parent,k)]
            del self.edgeWeightDict[(parent,k)]

            
        for node in fathernodes:
            i = self.queryGraphHeadList[node]
            while i != -1:
                if self.queryGraphHeadEdges[i][0]==k:
                    self.queryGraphHeadEdges[i][0] = j
                    
                i = self.queryGraphHeadEdges[i][1]
        for node in sonnodes:
            i = self.queryGraphTailList[node]
            while i!=-1:
                if self.queryGraphTailEdges[i][0]==k:
                    self.queryGraphTailEdges[i][0] = j
                i = self.queryGraphTailEdges[i][1]

        if self.queryGraphHeadList[k] != -1:  
            last_edge_index = self.queryGraphHeadList[j]
            if last_edge_index == -1:  
                self.queryGraphHeadList[j] = self.queryGraphHeadList[k]
            else:
                while self.queryGraphHeadEdges[last_edge_index][1] != -1:
                    last_edge_index = self.queryGraphHeadEdges[last_edge_index][1]
                # 将v的第一条边连接到u的最后一条边上
                self.queryGraphHeadEdges[last_edge_index][1] = self.queryGraphHeadList[k]

            # 最后，将v的head设置为-1，表示v不再是独立的节点
            self.queryGraphHeadList[k] = -1

        if self.queryGraphTailList[k] != -1:  # 检查v是否有出边
            last_edge_index = self.queryGraphTailList[j]
            if last_edge_index == -1:  # 如果u没有出边，直接连接
                self.queryGraphTailList[j] = self.queryGraphTailList[k]
            else:
                # 如果u有出边，找到u的最后一条边
                while self.queryGraphTailEdges[last_edge_index][1] != -1:
                    last_edge_index = self.queryGraphTailEdges[last_edge_index][1]
                # 将v的第一条边连接到u的最后一条边上
                self.queryGraphTailEdges[last_edge_index][1] = self.queryGraphTailList[k]

            # 最后，将v的head设置为-1，表示v不再是独立的节点
            self.queryGraphTailList[k] = -1
    '''
    图遍历方法及可替换计算
    '''
    def forward(self,operate,start_nodes,indegree_list=[],process_num=1):

        '''
        自前向后遍历全图，可传入操作方法
        '''
        if indegree_list==[]:
            indegree_list = [0]*self.totalNodes
            for _, link_target in self.edgeWeightDict:
                indegree_list[link_target] += 1
        new_start=set()
        for node in start_nodes:
            if node!='x':
                if indegree_list[node]==0:
                    new_start.add(node)
        queryGraphHeadList = self.queryGraphHeadList
        queryGraphHeadEdges = self.queryGraphHeadEdges
        q = deque(new_start)    

        while q:
            start_node = q.pop()
            operate(start_node)
            i = queryGraphHeadList[start_node]
            while i != -1:
                indegree_list[queryGraphHeadEdges[i][0]]-=1
                if indegree_list[queryGraphHeadEdges[i][0]]==0:
                    q.append(queryGraphHeadEdges[i][0])
                i = queryGraphHeadEdges[i][1]
    def backward(self,operate,start_nodes,outdegree_list=[]):
        '''
        自后向前遍历全图，可传入操作方法
        '''
        if outdegree_list==[]:
            outdegree_list = [0]*self.totalNodes
            for link in self.edgeWeightDict.keys():
                outdegree_list[link[0]]+=1
        new_start=[]
        for node in start_nodes:
            if node!='x':
                if outdegree_list[node]==0:
                    new_start.append(node)
        queryGraphTailList = self.queryGraphTailList
        queryGraphTailEdges = self.queryGraphTailEdges
        q = new_start
        while q:
            start_node = q.pop()
            operate(start_node)
            i = queryGraphTailList[start_node]
            while i != -1:
                outdegree_list[queryGraphTailEdges[i][0]]-=1
                if outdegree_list[queryGraphTailEdges[i][0]]==0:
                    # operate(queryGraphTailEdges[i][0])
                    q.append(queryGraphTailEdges[i][0])
                i = queryGraphTailEdges[i][1]
    
    def get_state_range_forward(self,node):
        '''
        用于构建参考坐标时，从父节点中获得坐标信息并计算本节点坐标
        '''
        parentnodes = self.findParentNodes(node)
        coordinates = [self.ref_coor[n][0] for n in parentnodes]
        coordinates.append(self.ref_coor[node][0])
        self.ref_coor[node] = [max(coordinates),self.ref_coor[node][1]]
    def get_state_range_backward(self,node):
        '''
        用于构建参考坐标时，从子节点中获得坐标信息并计算本节点坐标
        '''
        childrennodes = self.findChildNodes(node)
        coordinates = [self.ref_coor[n][1] for n in childrennodes]
        coordinates.append(self.ref_coor[node][1])

        self.ref_coor[node] = [self.ref_coor[node][0],min(coordinates)]

    def get_coordinate_forward(self,node):
        '''
        用于构建（前向）拓扑序坐标时，从父节点中获得坐标信息并计算本节点坐标
        '''
        parentnodes = self.findParentNodes(node)
        coordinates = [self.coordinateList[n] for n in parentnodes]
        coordinates.append(0)
        self.coordinateList[node] = max(coordinates)+1

    def get_coordinate_backward(self,node):
        '''
        用于构建（后）拓扑序坐标时，从子节点中获得坐标信息并计算本节点坐标
        '''
        childrennodes = self.findChildNodes(node)
        coordinates = [self.backwardCoordinateList[n] for n in childrennodes]
        coordinates.append(0)
        self.backwardCoordinateList[node] = max(coordinates)+1

    '''
    图信息计算方法
    '''

    def get_possible_path_forward(self,node):
        'free'
        '''
        用于构建（前向）拓扑序坐标时，从父节点中获得坐标信息并计算本节点坐标
        '''
        if self.path_num_list[node]==0:
            if self.findParentNodes(node)==[]:
                self.path_num_list[node]=1
            else:
                self.path_num_list[node]=0
                tempnum=0
                for fnode in self.findParentNodes(node):                        
                    tempnum+=self.path_num_list[fnode]
                self.path_num_list[node]=tempnum
    

    def calculateCoordinates(self):
        '''
        构建前向拓扑坐标
        '''
        
        self.coordinateList = [1]*self.totalNodes
        self.forward(self.get_coordinate_forward,self.startNodeSet)
        

    def calculateCoordinates_backward(self):
        '''
        构建后向拓扑坐标
        '''
        
        self.backwardCoordinateList = [1]*self.totalNodes
        self.backward(self.get_coordinate_backward,self.endNodeSet) 
        

    def local_update_coordinate(self,startnodes):
        '''
        部分更新拓扑拓扑坐标
        '''
        indegree_list = self.calculateDescendantNodeInDegree(startnodes)
        self.push_coordinate(set(startnodes),indegree_list)

    
    def calculateStateRange(self,reflist,mode='hmm'):
        '''
        计算参考坐标范围
        '''
        indegree_list = [0]*self.totalNodes
        outdegree_list = [0]*self.totalNodes
        for link in self.edgeWeightDict.keys():
            indegree_list[link[1]]+=1
            outdegree_list[link[0]]+=1
        self.ref_coor=[[0,len(reflist)]]*self.totalNodes
        for index,node in enumerate(reflist):
            if node != 'x':
                self.ref_coor[node] = [index,index] 

        self.forward(self.get_state_range_forward,self.startNodeSet,indegree_list)

        self.backward(self.get_state_range_backward,self.endNodeSet,outdegree_list)
        if mode=='build':
            pass
        if mode=='hmm':
            new_range=[]
            for node in range(self.totalNodes):
                childrens = self.findChildNodes(node)
                rights=[self.ref_coor[n][1] for n in childrens]
                rights.append(self.ref_coor[node][1])
                new_range.append([self.ref_coor[node][0],max(rights)])
            self.ref_coor = new_range


    def calculateReferenceCoordinates(self):
        self.calculateCoordinates_backward()
        head_nodes = self.findDescendantNodes(self.startNodeSet,self.longestPathNodeSet)

        coor_node_dict={}
        for node in self.longestPathNodeSet:
            if node in head_nodes:
                print('head_node')
                coor = self.maxlength - self.backwardCoordinateList[node]+1
            else:
                coor = self.coordinateList[node]
            coor_node_dict[coor] = coor_node_dict.get(coor,[])
            coor_node_dict[coor].append(node)
        return coor_node_dict
    
    def convertToAliReferenceDAG(self,thr,new_fra=20):
        '''
        查找参考图
        '''
        min_seq_num = max(thr*self.sequenceNum,1)
        for node in self.nodeList:
            node[3]=[node[0]]
        self.removeDegenerateBasePaths()
        node_set=set()
        for node in self.nodeList:
            if node[2]>min_seq_num:
                node_set.add(node[0])
        self.findLongestPath()
        self.calculateCoordinates_backward()
        now = self.endNodeSet
        doneset=set()
        while now !=set():
            next_set=set()
            for node in now:
                if self.nodeList[node][2]<0.01*self.sequenceNum and node not in doneset:
                    node_set-=set([node])
                    next_set.add(node)
                    doneset.add(node)

            now = set(self.findParentNodes(next_set))
        delset = set(range(self.totalNodes))-node_set
        self.edgeWeightDict = {link:value for link,value in self.edgeWeightDict.items() if not set(link) & delset}
        self.fast_remove(delset)
        self.reorder_ref_graph_nodes(node_set)
        self.findLongestPath()
        self.merge_check(mode='ali',coordDistanceLimit=10)
        

        startnodes=self.startNodeSet&self.longestPathNodeSet
        startNode=-1
        max_num=0
        for node in startnodes:
            if self.nodeList[node][2]>max_num:
                startNode=node
                max_num=self.nodeList[node][2]
        coor_cursor=1
        ref_seq = self.nodeList[startNode][1]
        ref_nodelist = [self.nodeList[startNode][3][0]]
        tmp_refnodelist = [startNode]
        now = set(self.findChildNodes(startNode))&self.longestPathNodeSet
        while now!=set():
            coor_cursor+=1
            max_num=0
            max_node = -1
            for node in now:
                if self.nodeList[node][2]>max_num and self.coordinateList[node]==coor_cursor:
                    max_node=node
                    max_num=self.nodeList[node][2]
            if max_node!=-1:
                ref_seq+=self.nodeList[max_node][1][-1]
                tmp_refnodelist.append(max_node)
                ref_nodelist.append(self.nodeList[max_node][3][0])
                now = set(self.findChildNodes(max_node))&self.longestPathNodeSet
            else:
                now=set()

        
        self.mergeNodesWithSameCoordinates(new_fra,commonBaseTypeThreshold=-1)
        self.fragmentReduce(1)
        self.findLongestPath()
        coor_node_dict = self.calculateReferenceCoordinates()

        emProbMatrix=np.full((4,self.maxlength),0,dtype=np.float64)
        # print(self.maxlength)
        # print(ref_seq)
        
        # exit()
        Adict={'A':[0],'T':[1],'C':[2],'G':[3],'R': [0, 3], 'Y': [2, 1], 'M': [0, 2], 'K': [3, 1], 'S': [3, 2], 'W': [0, 1], 'H': [0, 1, 2], 'B': [3, 1, 2], 'V': [3, 0, 2], 'D': [3, 0, 1], 'N': [0, 1, 2, 3]}
        for i in range(1,self.maxlength+1):
            for node in coor_node_dict[i]:
                bases = Adict[self.nodeList[node][1][-1]]
                for base in bases:
                    emProbMatrix[base][i-1]+=self.nodeList[node][2]/len(bases)
        add_emProbMatrix = np.full((4,self.fragmentLength-1),0,dtype=np.float64)
        addseq = ref_seq[:self.fragmentLength-1]
        for i in range(self.fragmentLength-1):
            bases = Adict[addseq[i]]
            for base in bases:
                add_emProbMatrix[base][i]+= 1/len(bases)
        
        emProbMatrix = np.hstack((add_emProbMatrix,emProbMatrix))
        sum_of_emProbMatrix = np.sum(emProbMatrix,axis=0)
        emProbMatrix = emProbMatrix/sum_of_emProbMatrix
        print(len(ref_seq),len(ref_nodelist))
        return ref_seq,ref_nodelist,emProbMatrix

    
    
    def findMainPathNodes(self):
        '''
        查找参考路径
        '''
        node_weight_list=[]
        for node in self.nodeList:
            if node[2]>self.sequenceNum/2:
                node_weight_list.append([node[0],self.coordinateList[node[0]]])
        sorted_list = sorted(node_weight_list, key=lambda x: x[1])
        reflist = [x[0] for x in sorted_list]
        return reflist
    
    '''
    图查询方法
    '''
    def findChildNodes(self, nodes):
        '''
        查找节点的子节点（仅查找本节点）
        '''
        if type(nodes)==int:
            nodes=[nodes]

        children = set()
        for node in nodes:
            i = self.queryGraphHeadList[node]
            while i != -1:
                children.add(self.queryGraphHeadEdges[i][0])
                i = self.queryGraphHeadEdges[i][1]
        return list(children)
    
    def findParentNodes(self,nodes):
        '''
        查找节点的父节点（仅查找本节点）
        '''
        if type(nodes)==int:
            nodes=[nodes]
        parent = set()
        for node in nodes:
            i = self.queryGraphTailList[node]
            while i!= -1:
                parent.add(self.queryGraphTailEdges[i][0])
                i = self.queryGraphTailEdges[i][1]
        return list(parent)
    

    def findChildrenClique(self, nodes):
        '''
        查找节点的子节点（仅查找本节点）
        '''
        if type(nodes)==int:
            nodes=[nodes]

        children = set()
        for node in nodes:
            i = self.headlist_clique[node]
            while i != -1:
                children.add(self.edges_clique[i][0])
                i = self.edges_clique[i][1]
        return list(children)
    def findParentClique(self,nodes):
        '''
        查找节点的父节点（仅查找本节点）
        '''
        if type(nodes)==int:
            nodes=[nodes]
        parent = set()
        for node in nodes:
            i = self.taillist_clique[node]
            while i!= -1:
                parent.add(self.tail_edges_clique[i][0])
                i = self.tail_edges_clique[i][1]
        return list(parent)


    def findDescendantNodes(self,startNode='',bannedSet=set()):
        '''
        查找子节点集
        '''
        now = set()
        if startNode=='':
            now = self.startNodeSet
        elif type(startNode)==list:
            now = set(startNode)
        elif type(startNode)==set:
            now = startNode.copy()
        else:
            now = set([startNode])
        
        finished=set()
        done_list=[0]*self.totalNodes
        for node in bannedSet:
            done_list[node]=1
        q = now
        while q:
            start_node = q.pop()
            
            if done_list[start_node]==0:
                i = self.queryGraphHeadList[start_node]
                while i != -1:
                    q.add(self.queryGraphHeadEdges[i][0])
                    i = self.queryGraphHeadEdges[i][1]
                done_list[start_node]=1
                finished.add(start_node)
        finished -= bannedSet
        return finished
    def findAncestorNodes(self,startNode='',bannedSet=set()):
        '''
        查找父节点集
        '''
        now = set()

        if startNode=='':
            for node in self.nodeList:
                if self.coordinateList[node[0]]==1:
                    now.add(node[0])
        elif type(startNode)==list:
            now = set(startNode)
        elif type(startNode)==set:
            now = startNode.copy()
        else:
            now = set([startNode])
        done_list=[0]*self.totalNodes
        for node in bannedSet:
            done_list[node]=1
        finished=set()
        queryGraphTailList = self.queryGraphTailList
        queryGraphTailEdges = self.queryGraphTailEdges
        q = now
        while q:
            start_node = q.pop()
            if done_list[start_node]==0:
                i = queryGraphTailList[start_node]
                while i != -1:
                    q.add(queryGraphTailEdges[i][0])
                    i = queryGraphTailEdges[i][1]
                done_list[start_node]=1
                finished.add(start_node)
        finished -= bannedSet
        return finished
    
    def findConsecutiveAncestorNodes(self,startNode='',bannedSet=set()):
        '''
        查找坐标连续父节点集
        '''
        now = set()

        if startNode=='':
            for node in self.nodeList:
                if self.coordinateList[node[0]]==1:
                    now.add(node[0])
        elif type(startNode)==list:
            now = set(startNode)
        else:
            now = set([startNode])
        finished=set()
        done_list=[0]*self.totalNodes
        for node in bannedSet:
            done_list[node]=1
        queryGraphTailList = self.queryGraphTailList
        coordinateList = self.coordinateList
        queryGraphTailEdges = self.queryGraphTailEdges
        q = now
        while q:
            start_node = q.pop()

            
            if done_list[start_node]==0:
                i = queryGraphTailList[start_node]
                stnode_coor = coordinateList[start_node]
                while i != -1:
                    if (stnode_coor-1)==coordinateList[queryGraphTailEdges[i][0]]:
                        q.add(queryGraphTailEdges[i][0])
                    i = queryGraphTailEdges[i][1]
                done_list[start_node]=1
                finished.add(start_node)

        return finished
    def calculateDescendantNodeInDegree(self,startNode=''):
        '''
        计算某节点所有子节点的入度
        '''
        indegree_list=[0]*self.totalNodes
        now = set()
        if startNode=='':
            for node in self.nodeList:
                if self.coordinateList[node[0]]==1:
                    now.add(node[0])
        elif type(startNode)==list:
            now = set(startNode)
        else:
            now = set([startNode])
        
        done_list=[0]*self.totalNodes
        q = now
        while q:
            start_node = q.pop()
            
            if done_list[start_node]==0:
                i = self.queryGraphHeadList[start_node]
                while i != -1:
                    indegree_list[self.queryGraphHeadEdges[i][0]]+=1
                    q.add(self.queryGraphHeadEdges[i][0])
                    i = self.queryGraphHeadEdges[i][1]
                done_list[start_node]=1

        return indegree_list
    
    def findConsecutiveDescendantNodes(self,startNode='',bannedSet=set()):

        '''
        查找坐标连续子节点集
        '''
        
        now = set()

        if startNode=='':
            for node in self.nodeList:
                if self.coordinateList[node[0]]==1:
                    now.add(node[0])
        elif type(startNode)==list:
            now = set(startNode)
        else:
            now = set([startNode])
        
        coordinateList = self.coordinateList
        queryGraphHeadList = self.queryGraphHeadList
        queryGraphHeadEdges = self.queryGraphHeadEdges
        done_list=[0]*self.totalNodes
        for node in bannedSet:
            done_list[node]=1
        finished=set()
        q = now
        while q:
            start_node = q.pop()
            
            if done_list[start_node]==0:
                i = queryGraphHeadList[start_node]
                while i != -1:
                    if (coordinateList[start_node]+1)==coordinateList[queryGraphHeadEdges[i][0]]:
                        q.add(queryGraphHeadEdges[i][0])
                    i = queryGraphHeadEdges[i][1]
                done_list[start_node]=1
                finished.add(start_node)

        return finished



    def findDescendantNodes_with_endindex(self,endindex,startNode=''):
        '''
        查找子节点集（范围内查找）
        '''
        
        now = set([startNode])
        done_list=[0]*self.totalNodes
        finished=set()  
        coordinateList = self.coordinateList
        headlsit = self.queryGraphHeadList
        queryGraphHeadEdges = self.queryGraphHeadEdges
        q = now
        while q:
            start_node = q.pop()
            if done_list[start_node]==0:
                i = headlsit[start_node]
                start_node_coor = coordinateList[queryGraphHeadEdges[i][0]]
                while i != -1:
                    if start_node_coor<=endindex:
                        q.add(queryGraphHeadEdges[i][0])
                    i = queryGraphHeadEdges[i][1]
                done_list[start_node]=1
                finished.add(start_node)
        return finished

    

    def nonCyclic(self,check_node,endindex,startNode=''):
        
        '''
        用于查找两个节点融合是否会导致图出现环结构
        '''
        without_loop=True
        now = set([startNode])
        done_list=[0]*self.totalNodes
        q = now
        queryGraphHeadList = self.queryGraphHeadList
        queryGraphHeadEdges = self.queryGraphHeadEdges
        coordinateList = self.coordinateList
        while q:
            start_node = q.pop()
            
            if done_list[start_node]==0:
                i = queryGraphHeadList[start_node]
                while i != -1:
                    if coordinateList[queryGraphHeadEdges[i][0]]<endindex:
                        q.add(queryGraphHeadEdges[i][0])
                    elif queryGraphHeadEdges[i][0]==check_node:
                        without_loop=False

                        return without_loop
                    i = queryGraphHeadEdges[i][1]
                done_list[start_node]=1

        return without_loop

    
    
    def push_coordinate(self,start_nodes,indegree_list=[],process_num=1):
        
        if indegree_list==[]:
            indegree_list = [0]*self.totalNodes
            for _, link_target in self.edgeWeightDict:
                indegree_list[link_target] += 1
        new_start=set()
        for node in start_nodes:
            if node!='x':
                if indegree_list[node]==0:
                    new_start.add(node)
        queryGraphHeadList = self.queryGraphHeadList
        queryGraphHeadEdges = self.queryGraphHeadEdges
        q = set(new_start)    
        tmpq=set()
        pushing=False
        while q:
            start_node = q.pop()
            parentnodes = self.findParentNodes(start_node)
            coordinates = [self.coordinateList[n] for n in parentnodes]
            coordinates.append(0)
            newcoor = max(coordinates)+1
            if newcoor!=self.coordinateList[start_node]:
                self.coordinateList[start_node] = newcoor
                pushing=True


            i = queryGraphHeadList[start_node]
            while i != -1:
                indegree_list[queryGraphHeadEdges[i][0]]-=1
                if indegree_list[queryGraphHeadEdges[i][0]]==0:
                    tmpq.add(queryGraphHeadEdges[i][0])
                i = queryGraphHeadEdges[i][1]
            if not q:
                if pushing==False:
                    break
                else:
                    q|=tmpq
                    tmpq=set()
                    pushing=False

        

    
    def coarse_grained_graph_construction(self,indegree_list=[]):
        
        '''
        将图划分为无分支节点的最大块
        '''
        
        if indegree_list==[]:
            indegree_list = [0]*self.totalNodes
            outdegree_list_static=[0]*self.totalNodes
            for outdegree, indegree in self.edgeWeightDict:
                indegree_list[indegree] += 1
                outdegree_list_static[outdegree]+=1
        indegree_list_static=indegree_list.copy()
        new_start=set()
        for node in self.startNodeSet:
            if node!='x':
                if indegree_list[node]==0:
                    new_start.add(node)

        q = deque(new_start)    

        clique_list=[]
        tmp_lst=[]


        while q:
            start_node = q.pop()
            tmp_lst=[start_node]
            while outdegree_list_static[start_node]==1:
                start_node = self.findChildNodes(start_node)[0]
                if indegree_list_static[start_node]==1:
                    tmp_lst.append(start_node)
                else:
                    break
                    

            clique_list.append(tmp_lst)
            for i in self.findChildNodes(tmp_lst[-1]):
                indegree_list[i]-=1
                if indegree_list[i]==0:
                    q.append(i)

        tail_dict={}

        for i in range(len(clique_list)):
            tail_dict[clique_list[i][-1]] = i

        clique_link=[]
        for i in range(len(clique_list)):
            for father_node in self.findParentNodes(clique_list[i][0]):
                if father_node == 109643:
                    print(clique_list[i][0],clique_list[i])
                f_clique = tail_dict[father_node]
                clique_link.append([f_clique,i])

        self.clique_num = len(clique_list)
        self.headlist_clique = [-1]*len(clique_list)
        self.taillist_clique = [-1]*len(clique_list)
        self.edges_clique=[]
        self.tail_edges_clique=[]
        for i in clique_link:
            self.edges_clique.append([i[1], self.headlist_clique[i[0]],0])
            self.headlist_clique[i[0]] = len(self.edges_clique) - 1
            self.tail_edges_clique.append([i[0], self.taillist_clique[i[1]],0])
            self.taillist_clique[i[1]] = len(self.tail_edges_clique) - 1

        self.startnodeset_clique = []
        self.endnodeset_clique = []
        for i in range(len(clique_list)):
            if self.findParentClique(i)==[]:
                self.startnodeset_clique.append(i)
            elif self.findChildrenClique(i)==[]:
                self.endnodeset_clique.append(i)
        
        return clique_list,clique_link
    


def load_graph(graphpath,newsavepath='',mode='without_onm',mindex=0,kill_degenerage=False,ziped=False):
    '''
    从文件中加载图
    '''
    # print('load graph '+graphpath)
    # 加载图文件
    graph_file = open(graphpath+'graph.pkl', 'rb')
    # 还原图结构
    graph = pickle.load(graph_file)
    # 加载图信息文件
    graph_data = np.load(graphpath+'data.npz',allow_pickle=True)
    # 还原图信息结构
    graph.nodeList = graph_data['nodeList'].tolist()
    graph.totalNodes = len(graph.nodeList)
    graph.coordinateList = graph_data['coordinateList'].tolist()
    graph.maxlength = max(graph.coordinateList)
    graph.edgeWeightDict = graph_data['edgeWeightDict']
    graph.fragmentNodeDict={}
    for node in  graph.nodeList:
        graph.fragmentNodeDict.setdefault(node[1], []).append(node[0]) # graph.fragmentNodeDict √ dict
    graph.longestPathNodeSet = set(graph_data['longestPathNodeSet'].tolist()) # graph.longestPathNodeSet √ set
    graph.startNodeSet = set(graph_data['startNodeSet'].tolist()) # graph.startNodeSet √ set
    graph.endNodeSet = set(graph_data['endNodeSet'].tolist()) # graph.endNodeSet √ set
    if mindex == 0:
        # 还原前向星结构
        graph.queryGraphHeadList = [-1]*graph.totalNodes
        graph.queryGraphTailList = [-1]*graph.totalNodes
        graph.queryGraphHeadEdges=[]
        graph.queryGraphTailEdges=[]
        
        graph.edgeWeightDict = { (link_weight[0],link_weight[1]):link_weight[2] for link_weight in graph.edgeWeightDict} # graph.edgeWeightDict √ dict
        for i in graph.edgeWeightDict.keys():
            graph.add_edge_to_queryGraph(i[0],i[1])


    if newsavepath!='':
        graph.savePath = newsavepath
    if mode!='without_onm':

        ori_node_list = np.load(graphpath+'onm.npy',allow_pickle=True)
        ori_node_index = np.load(graphpath+'onm_index.npy')
        for index,node in enumerate(graph.nodeList):
            oindex = ori_node_index[index]
            node[3]=ori_node_list[oindex[0]:oindex[1]]

    if kill_degenerage==True:
        graph.removeDegenerateBasePaths()

    graph_file.close()
    return graph


def build_graph(inpath,savePath,fragmentLength,maxExtensionLength=np.NINF,nodeIsolationThreshold=0,graphID='1'):
    
    '''
    extend_thr:主路径扩增阈值，若一次序列加入造成的主路径扩增大于阈值，则跳过该序列。若被跳过序列大于序列总数的0.005，则自动增加阈值
    nodeIsolationThreshold:存在参考路径范围大于该值，且权重低总序列数量的0.0001节点的序列，将被剔除。该参数为0表示不剔除任何序列
    '''
    starttime =datetime.now()
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    

    Sequence_record_list = SeqIO.parse(inpath, "fasta")
    seqs_list = []


    for sequence_record in Sequence_record_list:
        seqs_list.append(sequence_record)
    seq_num = len(seqs_list)
    
    def build(seqs_list,maxExtensionLength):
        start=0
        
        

        with alive_bar(seq_num,force_tty=True,length=20) as bar:
            for sequence_record in seqs_list:
                
                sequence = str(sequence_record.seq).upper()
                if start == 0:
                    graph = GRAPH(inpath,graphID, fragmentLength,savePath,sequence_record.id,sequence,maxExtensionLength=maxExtensionLength)
                    start=1
                else:
                    graph.add_seq(sequence,sequence_record.id)

                bar()


        print('The sequence addition has ended.')
        graph.map_to_OSM()
        graph.update_queryGraph()

        graph.findLongestPath()
        
        if nodeIsolationThreshold!=0:
            graph.remove_free_seq(nodeIsolationThreshold)
        
        print('start save')
        graph.save_graph(mode='build')
        endtime = datetime.now()
        print()
        print('extend_sp',graph.overExtensionSequences)
        print('continuosly_free',graph.lowMergeDegreeSequence)
        print('use time',endtime-starttime)
            


    
    build(seqs_list,maxExtensionLength)
    

