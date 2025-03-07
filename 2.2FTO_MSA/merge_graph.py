
import os
import sys
from datetime import datetime
from tqdm import tqdm
from Graph import *
import gc

# from sqlite_master import sql_master


def find_anchor_target(gp_base,gp_add,base_main,add_main):


    # 增图坐标列表
    Coordinate_list=[]
    # 锚定的包含同序列的两图间节点对集合（增图指向基图）
    tupset=set()

    base_seqs_dict = {}
    for node in base_main:
        base_seqs_dict[gp_base.nodeList[node[0]][1]] = base_seqs_dict.get(gp_base.nodeList[node[0]][1],[])
        base_seqs_dict[gp_base.nodeList[node[0]][1]].append(node[0])

    Coordinate_list=[]
    anchor_list=[]
    for node in add_main:
        
        seq = gp_add.nodeList[node[0]][1]
        optional_anchors = base_seqs_dict.get(seq,[])
        if len(optional_anchors)!=0:
            anchor = optional_anchors[0]
            Coordinate_list.append(gp_base.coordinateList[anchor])
            anchor_list.append((node[0],anchor,seq))

    blocklist,blockdif = GRAPH.array_to_block(Coordinate_list)
    print(blocklist,blockdif)
    # 查找坐标错误范围
    copy_rg =  GRAPH.Cyclic_Anchor_Combination_Detection(blocklist,blockdif)
    while copy_rg!=[]:
        for i in copy_rg:
            for j in range(i[0][1], i[1][1]+1):
                Coordinate_list[j] = -1
                anchor_list[j]=-1
        blocklist,blockdif = GRAPH.array_to_block(Coordinate_list)
        copy_rg =  GRAPH.Cyclic_Anchor_Combination_Detection(blocklist,blockdif)
    for anchor_tuple in anchor_list:
        if anchor_tuple!=-1:
            tupset.add(anchor_tuple)

    
    return tupset





def anchor_into_base_graph(gp_base,gp_add,newpath,anchor_tuple_list):
    print('anchor_into_base_graph')
    # 情况节点删除信息
    gp_base.dellist = []

    # 在基图中建立增图对应节点，并重定向原锚定节点的锚定信息
    newnodeset = set()
    for id in range(gp_add.totalNodes):
        
        if anchor_tuple_list[id]==-1:
            new_base_node = gp_base.add_new_node(gp_add.nodeList[id][1])        
            new_base_node[0] = gp_base.totalNodes
            newnodeset.add(new_base_node[0])
            gp_base.nodeList.append(new_base_node)
            gp_base.queryGraphHeadList.append(-1)
            gp_base.queryGraphTailList.append(-1)
            gp_base.totalNodes+=1
            gp_base.coordinateList.append('')
            anchor_tuple_list[id] = new_base_node[0]
            gp_base.fragmentNodeDict.setdefault(new_base_node[1],[]).append(new_base_node[0])
            
    
        
    # 转移增图节点信息
    for node in tqdm(gp_add.nodeList):
        tgnode = anchor_tuple_list[node[0]]
        gp_base.nodeList[tgnode][4] += node[4]
        if gp_base.nodeList[tgnode][4]:
            gp_base.startNodeSet.add(tgnode)
            gp_base.nodeList[tgnode][6]=1

        gp_base.nodeList[tgnode][5] += node[5]
        if gp_base.nodeList[tgnode][5]>0:
            gp_base.endNodeSet.add(tgnode)
            gp_base.nodeList[tgnode][6]=1



        
        gp_base.nodeList[tgnode][2] += node[2]
        gp_base.nodeList[tgnode][3] += node[3]

    for link in tqdm(gp_add.edgeWeightDict):
        try:
            gp_base.edgeWeightDict[(anchor_tuple_list[link[0]],anchor_tuple_list[link[1]])]+=link[2]
        except:
            gp_base.edgeWeightDict[(anchor_tuple_list[link[0]],anchor_tuple_list[link[1]])]=link[2]
    # gp_base.insert_link()
    gp_base.rebuild_queryGraph()
    del gp_add.edgeWeightDict
    gp_base.overExtensionSequences|=gp_add.overExtensionSequences
    gp_base.lowMergeDegreeSequence|=gp_add.lowMergeDegreeSequence
    gp_base.isolatedSequenceList|=gp_add.isolatedSequenceList
    
    gp_base.sequenceNum += gp_add.sequenceNum
    gp_base.savepath = newpath

    return newnodeset
        
def init_graph(graphfile,tmp_gindex):
    graph = load_graph(graphfile,mindex=0)
    for node in tqdm(graph.nodeList):
        node[3] = [(tmp_gindex,node[0])]
    graph.reflist = graph.findMainPathNodes()
    if tmp_gindex==1:
        newlinkset = []
        for i in graph.edgeWeightDict.items():
            newlinkset.append([i[0][0],i[0][1],i[1]])
        graph.edgeWeightDict = newlinkset

    weightnode_set=set()
    seq_num = graph.sequenceNum/2
    for i in graph.nodeList:
        if i[2]>seq_num and i[6]!=1:
            weightnode_set.add(i[0])

    coor_set=set()
    for weightnode in weightnode_set:
        coor = graph.coordinateList[weightnode]
        coor_set.add((weightnode,coor))
    
    main_list = sorted(coor_set, key=lambda x: x[1])
    return graph,main_list

def Tracing_merge(graphfileA,graphfileB,newpath):
    Trace_path = np.load(newpath+'/Traceability_path.npy',allow_pickle=True)
    
    source_A = np.load(graphfileA+'onm.npy',allow_pickle=True)
    print(source_A.shape)
    index_A = np.load(graphfileA+'onm_index.npy')
    source_B = np.load(graphfileB+'onm.npy',allow_pickle=True)
    index_B = np.load(graphfileB+'onm_index.npy')
    new_onm = np.full(source_A.size+source_B.size,0,dtype=object)
    new_index = []
    sources = [source_A,source_B]
    index_sources = [index_A,index_B]

    index_cursor=0
    # new_nodeid=0
    for cource_nodeids in tqdm(Trace_path):
        
        source_gid,source_nid = cource_nodeids[0]
        nid_index = index_sources[source_gid][source_nid]
        # print(nid_index)
        onms_list=[sources[source_gid][nid_index[0]:nid_index[1]]]
        # print(sources[source_gid][nid_index[0]:nid_index[1]+2])
        for source_nodeid in cource_nodeids[1:]:
            source_gid,source_nid = source_nodeid
            nid_index = index_sources[source_gid][source_nid]
            onms_list.append(sources[source_gid][nid_index[0]:nid_index[1]])

        onm = np.hstack(onms_list) 
        index = [index_cursor]
        index_cursor+=onm.size
        index.append(index_cursor)
        new_index.append(index)
        # print(index,onm.size,new_onm[index[0]:index[1]].size)
        new_onm[index[0]:index[1]] = onm
        # if onm.size==4:
        #     print(onm,cource_nodeids)
        #     exit()
    print(new_onm[-1])
    np.save(newpath+'onm.npy',new_onm)
    np.save(newpath+'onm_index.npy',np.array(new_index))
    print('onm saved')
    print('finished')
    v_idA = np.load(graphfileA+'v_id.npy')
    v_idB = np.load(graphfileB+'v_id.npy')
    np.save(newpath+'v_id.npy',list(v_idA)+list(v_idB))
    print('finished')

def Trace_zip(inpath,zippath):
    Trace_path = np.load(zippath+'/Traceability_path.npy',allow_pickle=True)
    
    source = np.load(inpath+'onm.npy',allow_pickle=True)
    index_list = np.load(inpath+'onm_index.npy')
    # new_onm = np.full(source.size,0,dtype=object)
    new_onm = []
    new_index = []
    index_cursor=0
    for cource_nodeids in tqdm(Trace_path):
        # print(cource_nodeids,'dddd')
        source_gid,source_nid = cource_nodeids[0]
        source_index = index_list[source_nid]
        onms_list = list(source[source_index[0]:source_index[1]])
        for source_nodeid in cource_nodeids[1:]:
            source_gid,source_nid = source_nodeid
            source_index = index_list[source_nid]
            # onms_list .append(source[source_index[0]:source_index[1]])
            onms_list.extend(list(source[source_index[0]:source_index[1]]))


        # onm = np.hstack(onms_list) 
        index = [index_cursor]
        index_cursor+=len(onms_list)
        index.append(index_cursor)
        new_index.append(index)
        # print(index,onm.size,new_onm[index[0]:index[1]].size)
        new_onm.extend(onms_list) 
        # new_nodeid+=1
    print('saving')
    np.save(zippath+'onm.npy',np.array(new_onm))
    np.save(zippath+'onm_index.npy',np.array(new_index))
    print('onm saved')
    print('finished')

def merge_graph(graphfileA,graphfileB,newpath):
    
    start = datetime.now()
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # 加载待合并子图
    graphA,main_list_A = init_graph(graphfileA,0)
    graphB,main_list_B = init_graph(graphfileB,1)

    print('start')
    print(graphfileA)
    print(graphfileB)

    # 寻找两图中的锚点，并根据锚点生成坐标列表和对应的节点列表
    anchored_pairs_A = find_anchor_target(graphA,graphB,main_list_A,main_list_B)
    anchored_pairs_B = find_anchor_target(graphB,graphA,main_list_B,main_list_A)

    anchor_tuple_list = [-1 for node in range(graphB.totalNodes)]
    # 根据互相指向的节点对重新写入锚定信息
    k=0
    for tup in tqdm(anchored_pairs_A):
        if (tup[1],tup[0],tup[2]) in anchored_pairs_B:
            anchor_tuple_list[tup[0]]=tup[1]
            k+=1

    anchor_into_base_graph(graphA,graphB,newpath,anchor_tuple_list)
    graphA.graphID =graphA.graphID+'&'+graphB.graphID
    graphA.fastaFilePathList.extend(graphB.fastaFilePathList)
    del graphB
    gc.collect()
    graphA.calculateCoordinates()
    graphA.updateCoordiante()
    print('end')
    print(graphA.maxlength)
    graphA.mergeNodesWithSameCoordinates(graphA.fragmentLength,commonBaseThreshold=0)
    print('update new main',len(graphA.longestPathNodeSet),len(graphA.nodeList))
    print(graphA.maxlength)

    graphA.currentEdgeSet = set()    

    graphA.save_graph(savepath=newpath,mode='merge')

    end = datetime.now()
    print('end',end-start)


if __name__ == '__main__':
    graphfileA = str(sys.argv[1])
    graphfileB = str(sys.argv[2])
    newpath = sys.argv[3]
    merge_graph(graphfileA,graphfileB,newpath)
    Tracing_merge(graphfileA,graphfileB,newpath)


