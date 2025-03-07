
import sys
import numpy as np
from Graph import build_graph

# biggraphid=str(sys.argv[1])
name = str(sys.argv[1])

inpath = '/state/lx_backup/sonset_of_Is_complete_230515_num_lowN/{}.fasta'.format(name)
# inpath = '/state1/DATA/new_far_test/1_500.fasta'.format(name)

outpath = '/state1/result_DATA_back_up/graph_of_Is_complete_230515/{}/'.format(name)
# outpath = 'tt/'.format(name)
fra = 64
print(inpath,outpath,fra)
g_name = name.split('_')[0]
build_graph(inpath,outpath,fra,maxExtensionLength=1000,graphID=name,nodeIsolationThreshold=1000)
# build_graph(inpath,savePath,FragmentLength,maxExtensionLength=np.NINF,nodeIsolationThreshold=0,graphID='1')
# build_graph(inpath,outpath,fra,extend_thr=160,graph_name=g_name,rm_free_length=0)
