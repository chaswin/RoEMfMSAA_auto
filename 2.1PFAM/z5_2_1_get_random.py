# 是这样的：sp和me根据序列数量的增长而极具膨胀。我们不能直接将阶梯数据集所对应的hmmpx，sp，me放在一张图，这没有意义。
# 我们必须在每一个阶梯px所对应的数据集中，随机抽取跟最初的第0阶梯的数据集条数相等的数据集，才能去做sp和me的阶梯曲线。
# 本程序从0阶梯开始：取0文件夹内的seqac放入[]，随机抽取跟0中的fa一样的数据量，组成新seqs，生成新的fa放进1。（每个文件夹内的seqac和seqs对应的都是下一个文件夹的fa和hmmpx）
import os
import random
import re
import sys
import time
from Bio.SeqIO.FastaIO import SimpleFastaParser



def del_random(hmm_out_path, ac, i):

    # os.system('rm -r {hmm_out_path}/{ac}/{s}/random/'.format(hmm_out_path=hmm_out_path, ac=ac, s=+1))  # 删除random

    os.system('rm {hmm_out_path}/{ac}/{s}/*.fa'.format(hmm_out_path=hmm_out_path, ac=ac, s=i))  # 事后删除.fa，太大了几十个G，没用

    # os.system('rm -r {outpath}/{ac}/{s}/random/{ac}_PF00083_align_big.fam'
    #      .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=i+1))     # 删除只保留match部分的比对文件fam



def get_random(hmm_out_path, ac, s, initial_seqsnum):

    try:  # 创建一下，要是有就报错略过。
        os.makedirs(hmm_out_path + '/{ac}/{s}/random/'.format(ac=ac, s=s+1))  # 创建多级文件夹的命令。
    except:
        pass

    hmm_seqac = []  # 存放序列号的列表
    with open('{path}/{ac}/{s}/{ac}_{pf}_hmm_uni_seqac'.format(path=hmm_out_path, ac=ac, s=s, pf=pf)) as f:  # 打开out
        while 1:  # 遍历每一行
            line = f.readline()  # 按行读取避免内存溢出
            if not line:
                break            # 遇到空白行，退出
            if line[0] != '#':   # 只要不是注释行
                hmm_seqac.append(line)  # 没退出就继续执行，把序列号存在列表
        f.close()

    # # 其实这一步可有可无，不一定非要跟文件0里面的序列的数量保持一致。我们其实可以都采用100条，除非本家族前期迭代少于100条
    # with open('{path}/{ac}/0/{ac}_{pf}.hmm'.format(path=hmm_out_path, ac=ac, pf=pf)) as f:
    #     while 1:  # 遍历每一行
    #         line = f.readline()  # 按行读取避免内存溢出
    #         if not line:
    #             break           # 遇到空白行，退出
    #         if line[0:4] == 'NSEQ':   # 只要不是注释行
    #             initial_seqsnum = line.split()[1]  # 没退出就继续执行，把序列号存在列表
    #             # print(initial_seqsnum)
    #     f.close()

    # # 4.2把序列号写入txt
    # with open('{path}/{ac}/{s}/random/{ac}_PF00083_hmm_uni_seqac'.format(path=hmm_out_path, ac=ac, s=s+1), 'w') as o:
    #     # 所以这一步的 int(initial_seqsnum) 可以直接换成数字100
    #     # random_seqac = random.sample(hmm_seqac, int(initial_seqsnum))
    #     random_seqac = random.sample(hmm_seqac, 100)
    #     for seqac in random_seqac:
    #         o.write(seqac)
    #     o.close()
    # 4.2但是呢，由于随机抽取序列的效果实际上不太好，决定按照序列号的顺序（本身就是相似度越来越远的排序），拿取100条。或者是initial条。
    with open('{path}/{ac}/{s}/random/{ac}_{pf}_hmm_uni_seqac'.format(path=hmm_out_path, ac=ac, s=s+1, pf=pf), 'w') as o:
        hmm_seqac_num = len(hmm_seqac)                    # 总共的序列号数目
        print(hmm_seqac_num)
        
        initial_seqsnum = int(initial_seqsnum)
        if initial_seqsnum >= 100:
            step = int(round(hmm_seqac_num / 100, 0))         # 你将要在hmm_seqac_num个序列号中取100条，因此步长是每隔step取。
        else:
            step = int(round(hmm_seqac_num / initial_seqsnum, 0))
        print(step)
        
        s2 = 0
        for i in range(0, hmm_seqac_num, step):
            o.write(hmm_seqac[i])
            s2 += 1
            if s2 == 100: break                            # 再判断一下，否则总是有误差
        o.close()

    # 5.提取序列，同上一个文件的步骤（三）用blast+去提取序列们，合成一个fasta
    os.system('blastdbcmd -db {db} \
                          -entry_batch {inpath}/{ac}/{s}/random/{ac}_{pf}_hmm_uni_seqac>\
                          {outpath}/{ac}/{s}/random/{ac}_{pf}_hmm_uni_seqs'
              .format(db=db, inpath=hmm_out_path, outpath=hmm_out_path, ac=ac, s=s+1, pf=pf))



    # 备用搞事！！！！

    # 方案一：mafft(直接给random生成多序列比对fa)
    os.system('mafft --retree 1 --maxiterate 0 --nofft --parttree \
                     --quiet \
                     {inpath}/{ac}/{s}/random/{ac}_{pf}_hmm_uni_seqs > {outpath}/{ac}/{s}/random/{ac}_{pf}.fa'
              .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1, pf=pf))
    # 为了后续自己给这些random再训练phmm，需要初始参数。这里用hmmbuild根据mafft的比对生成模型hmm，作为初始参数。
    os.system('hmmbuild \
              {outpath}/{ac}/{s}/random/{ac}_{pf}.hmm \
              {inpath}/{ac}/{s}/random/{ac}_{pf}.fa'
              .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1, pf=pf))

    # 方案二：hmmalign(用方案一mafft给random100条的比对fa生成的小hmm，反过来再次指导生成比对align.fa)
    os.system('hmmalign --outformat afa \
               -o {outpath}/{ac}/{s}/random/{ac}_{pf}_align.fa \
               {inpath}/{ac}/{s}/random/{ac}_{pf}.hmm {inpath}/{ac}/{s}/random/{ac}_{pf}_hmm_uni_seqs'
              .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1, pf=pf))

    # # 方案三：hmmalign(用外层完整大数据集的大hmm，指导生成比对align_big.fa)
    # os.system('hmmalign --outformat afa \
    #            -o {outpath}/{ac}/{s}/random/{ac}_PF00083_align_big.fa \
    #            {inpath}/{ac}/{s}/{ac}_PF00083.hmm {inpath}/{ac}/{s}/random/{ac}_PF00083_hmm_uni_seqs'
    #           .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1))

    # # 方案四：hmmbuild(用方案三的比对align_big.fa，重新构建相应的小模型align_big.hmm，看是否比大模型小但又不失去大模型特征)
    # os.system('hmmbuild \
    #           {outpath}/{ac}/{s}/random/{ac}_PF00083_align_big.hmm \
    #           {inpath}/{ac}/{s}/random/{ac}_PF00083_align_big.fa'
    #           .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1))

    # 备用搞事！！！！！
    # （只比较match部分的前置处理）读取文件，生成只有大写字母和删除部分的fasta。
    # 文件名每次都修改一下，看好是要对哪个文件这样做
    with open('{outpath}/{ac}/{s}/random/{ac}_{pf}_align.fa'
                      .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1, pf=pf)) as ff, \
        open('{outpath}/{ac}/{s}/random/{ac}_{pf}_align.fam'
                      .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s+1, pf=pf), 'w') as oo:
        for title, seq in SimpleFastaParser(ff):  # 对于该数据库中的每项中的，标题和序列。注意，title不包含'>'。
            ac = title.split()[0]  # 根据具体情况进行修改
            seq_match_li = re.findall(r'[A-Z-]', seq)   # 正则匹配所有的大写字母和-
            seq_match = ''
            for char in seq_match_li:                   # 把正则结果列表变成字符串
                seq_match += char
            # print(len(seq_match))                     # 放心，一样长
            oo.write('>{}\n{}\n'.format(ac + ' ' + title.split(' ', 1)[1], seq_match))  # 就写入oo
        ff.close()
        oo.close()



if __name__ == '__main__':
    local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    # local_path = '/data1/kzw'

    # pf = 'PF00083'
    pf = sys.argv[1]

    db = '{lp}/repfam/{pf}/{pf}_uniprot_seq/db/{pf}_uniprot_seqs'.format(lp=local_path, pf=pf)  # uni的blast数据库
    hmm_out_path = '{lp}/repfam/{pf}'.format(lp=local_path, pf=pf)  # 结果文件保存的地址

    # onefamily_seqacs = ['O97467']
    seqac = sys.argv[2]
    onefamily_seqacs = [seqac]

    times = int(sys.argv[3])
    # times = 41

    for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列

        # 最初的fa里面有多少条序列。后续都跟这个对齐。也就是，每个有结构的序列的第一次的fa的数量，后续都会跟这个数量同步。
        with open('{path}/{ac}/0/{ac}_{pf}.hmm'.format(path=hmm_out_path, ac=ac, pf=pf)) as f:
            while 1:  # 遍历每一行
                line = f.readline()  # 按行读取避免内存溢出
                if not line:
                    break  # 遇到空白行，退出
                if line[0:4] == 'NSEQ':  # 只要不是注释行
                    initial_seqsnum = line.split()[1]  # 最初的fa里面有多少条序列。
            f.close()

        for s in range(times):  # z4.2里做了几个times/节点/阶梯就range几。这里不需要+1，因为从0取出的seqs数据集会在1中生成random。
            print(time.ctime(), 'random开始在此处生成：', s+1)
            # del_random(hmm_out_path, ac, s)    # 慎选，删除整个random文件夹了
            get_random(hmm_out_path, ac, s, initial_seqsnum)
            print(time.ctime(), 'random在此处任务结束：', s+1)



# 关于本程序的另一种可能的写法，应用到如下两个软件。但，目前没用上。
# esl-selectn 10 '/home/handsomekk/repfam/PF00083/O97467/0/O97467_PF00083_hmm_uni_seqac' > 10
# blastdbcmd -db /home/handsomekk/repfam/PF00083/PF00083_uniprot_seq/db/PF00083_uniprot_seqs -entry_batch 10>10.fasta