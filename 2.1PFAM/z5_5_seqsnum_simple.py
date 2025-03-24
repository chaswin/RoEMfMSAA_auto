# 本程序提取每次迭代文件夹之内的，数据集大小。
# 简要版

import os
import sys
from Bio.SeqIO.FastaIO import SimpleFastaParser



if __name__ == '__main__':
    # local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    local_path = '/data1/kzw'

    # pf = 'PF00083'
    pf = sys.argv[1]

    db = '{lp}/repfam/{pf}/{pf}_uniprot_seq/db/{pf}_uniprot_seqs'.format(lp=local_path, pf=pf)  # uni的blast数据库
    hmm_out_path = '{lp}/repfam/{pf}'.format(lp=local_path, pf=pf)  # 结果文件保存的地址

    # onefamily_seqacs = ['O97467']
    seqac = sys.argv[2]
    onefamily_seqacs = [seqac]

    times = int(sys.argv[3])

    for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列

        # 每次迭代文件夹之内的，数据集大小
        seqsnum = ['seqsnum']
        for i in range(1, times+1):  # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。***************************
            with open('{path}/{ac}/{s}/{ac}_{pf}.hmm'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)) as f:
                while 1:  # 遍历每一行
                    line = f.readline()  # 按行读取避免内存溢出
                    if not line:
                        break  # 遇到空白行，退出
                    if line[0:4] == 'NSEQ':  # 只要不是注释行
                        seqsnum.append(line.split()[1])  # 对应的数据集里面有多少条序列。在hmm文件中寻找这个资料。
                f.close()
        with open(hmm_out_path + '/{ac}/get_seqsnum'.format(ac=ac), 'w') as o:
            for every_seqsnum in seqsnum:
                o.write(str(every_seqsnum) + '\n')
            o.close()

        # 分析所有数据集梯度：最外层hmmerhmm列数、文件大小变化
        hmmerhmm = ['hmmerhmm']
        for i in range(1, times+1):  # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。*****************************
            hmmerhmm_file = '{path}/{ac}/{s}/{ac}_{pf}.hmm'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            hmmerhmm_file_size = os.path.getsize(hmmerhmm_file)
            with open(hmmerhmm_file) as f:
                while 1:  # 遍历每一行
                    line = f.readline()  # 按行读取避免内存溢出
                    if line[0:4] == 'LENG':
                        hmmerhmm.append(line.split()[1] + ',' + str(hmmerhmm_file_size))   # 模型长度即match数量 + 文件大小
                        break
                f.close()
        with open(hmm_out_path + '/{ac}/get_hmmerhmm'.format(ac=ac), 'w') as o:
            for every_hmmerhmm in hmmerhmm:
                o.write(str(every_hmmerhmm) + '\n')
            o.close()

        # 分析所有数据集梯度random文件夹内的：fa文件大小、hmm列数、文件大小变化
        randomfileinfo = ['mafftfa,mafhmm_leng,mafhmm_size,alignfa,alignfam_leng']
        for i in range(1, times+1):  # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。*********************************
            mafftfa_file = '{path}/{ac}/{s}/random/{ac}_{pf}.fa'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            mafftfa_file_size = os.path.getsize(mafftfa_file)
            mafhmm_file = '{path}/{ac}/{s}/random/{ac}_{pf}.hmm'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            mafhmm_file_size = os.path.getsize(mafhmm_file)
            alignfa_file = '{path}/{ac}/{s}/random/{ac}_{pf}_align.fa'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            alignfa_file_size = os.path.getsize(alignfa_file)
            # alignbigfa_file = '{path}/{ac}/{s}/random/{ac}_{pf}_align_big.fa'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            # alignbigfa_file_size = os.path.getsize(alignbigfa_file)
            # alignbighmm_file = '{path}/{ac}/{s}/random/{ac}_{pf}_align_big.hmm'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            # alignbighmm_file_size = os.path.getsize(alignbighmm_file)
            # subfa_file = '{path}/{ac}/{s}/random/{ac}_{pf}_subop_vi.fa'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            # subfa_file_size = os.path.getsize(subfa_file)
            # vifa_file = '{path}/{ac}/{s}/random/{ac}_{pf}_vi.fa'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            # vifa_file_size = os.path.getsize(vifa_file)

            alignfam_file = '{path}/{ac}/{s}/random/{ac}_{pf}_align.fam'.format(path=hmm_out_path, ac=ac, s=i, pf=pf)
            with open(alignfam_file) as fm:
                for title, seq in SimpleFastaParser(fm):  # 对于该数据库中的每项中的，标题和序列。注意，title不包含'>'。
                    alignfam_file_leng = len(seq)        # .fam文件的match列的数量
                    break

            with open(mafhmm_file) as f:
                while 1:  # 遍历每一行
                    line = f.readline()  # 按行读取避免内存溢出
                    if line[0:4] == 'LENG':
                        mafhmmleng = line.split()[1]     # 模型长度即match数量
                        randomfileinfo.append(str(mafftfa_file_size) + ',' +
                                              mafhmmleng + ',' + str(mafhmm_file_size) + ',' +
                                              str(alignfa_file_size) + ',' + str(alignfam_file_leng))
                        break
                f.close()
        with open(hmm_out_path + '/{ac}/get_randomfileinfo'.format(ac=ac), 'w') as o:
            for every_randomfileinfo in randomfileinfo:
                o.write(str(every_randomfileinfo) + '\n')
            o.close()
