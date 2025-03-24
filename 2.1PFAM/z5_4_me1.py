# minimum entropy打分程序。针对random文件夹。输入文件格式为fasta。
# 思路：biopython读取器直接读取存为矩阵。先弄伪计数，再按照公式-Mcialogpia算每一列。

import collections
import time
import sys
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser      # fasta就用fasta，fastaq需要用另外的q



def aa_generator():    # 氨基酸残基生成器。
    aa_li = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B',
             'Z', 'X', '-']
    return aa_li



def me(align_file, aa_li):

    count_dic = {}  # 格式：{位置：{字符：数量， 字符：数量}， 位置：{字符：数量， 字符：数量}， …… }
    seqs_li = []  # 序列们的列表
    len_seq = 0  # 记录一下序列的长，也就是矩阵的宽度，矩阵的列数

    # （一）先用biopython读取。
    with open(align_file) as f:

        # 总之先想办法把序列存进数组矩阵中。
        # 此处我们将fasta格式的比对结果的文件，看做许多条带空格的序列。

        for title, seq in SimpleFastaParser(f):
            seq_li = []       # 每条单条序列再套一个列表
            for char in seq:     # 把一条序列中的每个字符
                if char == '.':
                    char = '-'
                seq_li.append(char.upper())      # 单独添加进这个序列的列表中     # seq_li的格式就是['-', '-', '-', '-', '-',
            seqs_li.append(seq_li)   # 把每条序列的列表再放进seqs_li  # seqs_li就变成了列表的嵌套 [['-', '-', ……,'-'], ['-', '-', ……, '-',]]
            len_seq = len(seq)

        seq_am = np.array(seqs_li)   # 把这样的列表嵌套放进nparray，就做成了矩阵。

        for c_position in range(len_seq):                   # 对于每一列来说
            count_result = collections.Counter(seq_am[:, c_position])   # 这个函数直接计算了该矩阵的第c_position的所有元素的出现次数
            # 注意不要这样写↓，直接像后面那样dict即可，否则字符串后面无法操作
            # print(str(count_result).split('(')[-1][0:-1])       # 做一下字符串切片 {'Q': 65, '-': 5, 'E': 1}
            count_dic[c_position] = dict(count_result)   # 添加进字典中，key是位置，value是如上具体的字符和数目

        f.close()   # 文件信息读取任务完成。

    ##############################################################################################################

    # print(len(seqs_li))  # 序列条数
    print(len_seq)       # 序列宽度
    # print(count_dic)     # 最终成品字典。

    #（二）计算
    scores = 0                # 所有总分
    for c_position in range(len_seq):        #  对于每一列执行计算

        # print(type(count_dic[c_position]))  # 这里面验证一下。现在是dict了，不是str。
        # print(count_dic[c_position])         # 每一列所对应的，所有元素的分别计数。   {'-': 70, 'M': 1}

        score = 0             # 单列分数

        cia_all = 0           # 也就是cia撇，该列的所有元素的总计数（包括伪计数）
        for aa in aa_li:
            if aa in count_dic[c_position]:
                cia_all += count_dic[c_position][aa]+1  # 有就计数
            else:
                cia_all += 0+1                          # 没有就伪计数

        for aa in aa_li:
            if aa in count_dic[c_position]:
                cia = count_dic[c_position][aa]+1
                pia = cia/cia_all
                s = -(cia * np.log(pia))
                score += s
            else:
                cia = 0+1
                pia = 1/cia_all
                s = -(cia * np.log(pia))
                score += s

        # print('每列分数：'+str(score))   # 每列分数
        scores += score

    # print(scores)
    # return scores
    return scores / len_seq

    # print(int(scores / len_seq))
    # return int(scores / len_seq)   # （X）不应该是算每列，而是算每条序列的，实际上。？

    # print(int(scores / len(seqs_li)))
    # return int(scores / len(seqs_li))   # 算每条序列的。


if __name__ == '__main__':
    # local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    local_path = '/data1/kzw'

    # pf = 'PF00083'
    pf = sys.argv[1]

    hmm_out_path = '{lp}/repfam/{pf}'.format(lp=local_path, pf=pf)  # 上一步的它们的结果文件保存的地址

    # onefamily_seqacs = ['G4TS85']  # 本次我只会用到这一个,在这个目前的83家族中，针对这一个做一下。后续可以通过for进行添加。
    seqac = sys.argv[2]
    onefamily_seqacs = [seqac]

    times = int(sys.argv[3])

    aa_li = aa_generator()

    # # 计算mafft的比对
    # for ac in onefamily_seqacs:        # 对于该家族中的所有的有结构序列
    #     me_li = ['me']     # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的me。并在开头事先存放一个字头。
    #     for i in range(1, 42):              # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'me开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         me_li.append(me(align_file, aa_li))
    #         print(time.ctime(), 'me结束', i)
    #     with open(hmm_out_path + '/{ac}/get_me'.format(ac=ac), 'w') as o:
    #         for every_me in me_li:
    #             o.write(str(every_me) + '\n')
    #         o.close()

    # 计算随机100条小hmm指导的比对
    for ac in onefamily_seqacs:        # 对于该家族中的所有的有结构序列
        me_li = ['me_align']     # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的me。并在开头事先存放一个字头。
        for i in range(1, times+1):              # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
            print(time.ctime(), 'me_align开始', i)
            align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_align.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
            me_li.append(me(align_file, aa_li))
            print(time.ctime(), 'me_align结束', i)
        with open(hmm_out_path + '/{ac}/get_me_align'.format(ac=ac), 'w') as o:
            for every_me in me_li:
                o.write(str(every_me) + '\n')
            o.close()
    # 计算随机100条小hmm指导的比对————只有match列版
    for ac in onefamily_seqacs:        # 对于该家族中的所有的有结构序列
        me_li = ['me_align_m']     # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的me。并在开头事先存放一个字头。
        for i in range(1, times+1):              # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
            print(time.ctime(), 'me_align_m开始', i)
            align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_align.fam'.format(lp=local_path, s=i, ac=ac, pf=pf)
            me_li.append(me(align_file, aa_li))
            print(time.ctime(), 'me_align_m结束', i)
        with open(hmm_out_path + '/{ac}/get_me_align_m'.format(ac=ac), 'w') as o:
            for every_me in me_li:
                o.write(str(every_me) + '\n')
            o.close()

    # # 计算完整大数据集的hmm指导的比对
    # for ac in onefamily_seqacs:        # 对于该家族中的所有的有结构序列
    #     me_li = ['me_align_big']     # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的me。并在开头事先存放一个字头。
    #     for i in range(1, 42):              # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'me_align_big开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_align_big.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         me_li.append(me(align_file, aa_li))
    #         print(time.ctime(), 'me_align_big结束', i)
    #     with open(hmm_out_path + '/{ac}/get_me_align_big'.format(ac=ac), 'w') as o:
    #         for every_me in me_li:
    #             o.write(str(every_me) + '\n')
    #         o.close()
    # # 计算我自己写的viterbi
    # for ac in onefamily_seqacs:        # 对于该家族中的所有的有结构序列
    #     me_li = ['me_vi']     # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的me。并在开头事先存放一个字头。
    #     for i in range(1, 42):              # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'me_vi开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_vi.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         me_li.append(me(align_file, aa_li))
    #         print(time.ctime(), 'me_vi结束', i)
    #     with open(hmm_out_path + '/{ac}/get_me_vi'.format(ac=ac), 'w') as o:
    #         for every_me in me_li:
    #             o.write(str(every_me) + '\n')
    #         o.close()
    # # 计算我自己写的次优vi
    # for ac in onefamily_seqacs:        # 对于该家族中的所有的有结构序列
    #     me_li = ['me_subop_vi']     # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的me。并在开头事先存放一个字头。
    #     for i in range(1, 42):              # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'me_sub开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_subop_vi.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         me_li.append(me(align_file, aa_li))
    #         print(time.ctime(), 'me_sub结束', i)
    #     with open(hmm_out_path + '/{ac}/get_me_subop_vi'.format(ac=ac), 'w') as o:
    #         for every_me in me_li:
    #             o.write(str(every_me) + '\n')
    #         o.close()