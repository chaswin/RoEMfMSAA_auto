# 这个相当于简易版的clustalo导出cluster.aux功能。所以也可以直接用它分好类的aux文件。参考我整理的aux精要
# 本文件从单一家族的stseqs生成的dnd树文件中，根据距离分cluster。0.0-0.3，0.3-0.5，0.5-0.7,0.7-1.0

from Bio import Phylo
# biopython1.82版第16章


def cluster_dnd_nclus(dnd_path, out_path):
    cluster1 = {}
    cluster2 = {}
    cluster3 = {}
    cluster4 = {}
    tree = Phylo.read(dnd_path, 'newick')
    # Phylo.draw_ascii(tree)
    li = tree.get_terminals()                 # [Clade(branch_length=0.362976, name='sp|O35956|S22A6_RAT'),
    # print(li[0].branch_length)              # 0.362976
    # print(li[0].name.split('|')[1])         # O35956
    for i in li:
        bl = i.branch_length  # float
        name = i.name.split('.')[0]  # str
        if 0<=bl<1:
            cluster1[name] = bl
        elif 0.3<=bl<0.5:
            cluster2[name] = bl
        elif 0.0014<=bl<0.01:
            cluster3[name] = bl
        elif 0.01<=bl<0.17:
            cluster4[name] = bl
        else:
            print("dnd文件中的距离不在分类范围内")

    with open(out_path, 'w') as o:
        for i in cluster1:
            o.write('cluster1,' + i + ',' + str(cluster1.get(i)) +'\n')
        for i in cluster2:
            o.write('cluster2,' + i + ',' + str(cluster2.get(i)) + '\n')
        for i in cluster3:
            o.write('cluster3,' + i + ',' + str(cluster3.get(i)) + '\n')
        for i in cluster4:
            o.write('cluster4,' + i + ',' + str(cluster4.get(i)) + '\n')
        o.close()

    return cluster1, cluster2, cluster3, cluster4


if __name__ == '__main__':
    # local_path = '/disk2/kzw'  # 我的路径
    local_path = '/home/handsomekk'  # 我的30pc
    pf = 'PF00083'  # 你的家族号
    dnd_path = '/home/handsomekk/repfam/PF00083/PF00083_stseq_ana/P11169/P11169_PF00083_bpseqs.dnd'  # 30pc测试文件
    out_path = '/home/handsomekk/repfam/PF00083/PF00083_stseq_ana/P11169/P11169_PF00083_bpseqs_dnd.cluster'  # 30pc测试文件
    # dnd_path = '{}/repfam/{}/{}_stseq_ana/{}_stseqs.dnd'.format(local_path, pf, pf, pf)
    # out_path = '{}/repfam/{}/{}_stseq_ana/{}_stseqs_dnd.cluster'.format(local_path, pf, pf, pf)
    cluster_dnd_nclus(dnd_path, out_path)