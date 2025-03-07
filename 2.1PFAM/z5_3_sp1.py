# 连续性相似梯度数据集的sp打分程序。针对random。其实只要是输入文件格式为fasta就可以。
# 思路：先用biopython读取器直接读取，记录每一个位置有多少什么字符，最后用组合公式，和简单的乘法。计算出每个组合有多少，再打分。

import collections
import itertools
import time
import numpy as np
import math
import sys
from Bio.SeqIO.FastaIO import SimpleFastaParser      # fasta就用fasta，fastaq需要用另外的q
from Bio.Align import substitution_matrices
matrix = substitution_matrices.load("BLOSUM62")
# matrix = substitution_matrices.load("BLOSUM50")



def combination_generator():    # 氨基酸组合生成器，包括了自身重复，共300组。
    aa_li = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B',
             'Z', 'X', '-']
    # itertools模块下combinations_with_replacementlist函数可以生成组合，且包括自身重复。
    c = list(itertools.combinations_with_replacement(aa_li,2))
    # print(c)       # 是正确的，仅仅是组合。而不包括定向的排列的两种。
    # print(len(c))  # list里包tuple [('A', 'R'), ('A', 'N'), ('A', 'D'), 一共300个。
    return c



def sp(align_file, c):

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
                seq_li.append(char.upper())   # 单独添加进这个序列的列表中     # seq_li的格式就是['-', '-', '-', '-', '-',
            seqs_li.append(seq_li)   # 把每条序列的列表再放进seqs_li  # seqs_li就变成了列表的嵌套 [['-', '-', ……,'-'], ['-', '-', ……, '-',]]
            len_seq = len(seq)

        seq_am = np.array(seqs_li)   # 把这样的列表嵌套放进nparray，就做成了矩阵。

        for c_position in range(len_seq):                   # 对于每一列来说
            count_result = collections.Counter(seq_am[:, c_position])   # 这个函数直接计算了该矩阵的第c_position的所有元素的出现次数
            # 注意不要这样写↓，直接像后面那样dict即可，否则字符串后面无法操作
            # print(str(count_result).split('(')[-1][0:-1])       # 做一下字符串切片 {'Q': 65, '-': 5, 'E': 1}
            count_dic[c_position] = dict(count_result)   # 添加进字典中，key是位置，value是如上具体的字符和数目

        f.close()   # 文件信息读取任务完成。

    ###############################################################################################################

    # print(len(seqs_li))  # 序列条数
    print(len_seq)       # 序列宽度
    # print(count_dic)     # 最终成品字典。

    #（二）计算
    scores = 0                # 所有总分
    for c_position in range(len_seq):        #  对于每一列执行计算

        # print(type(count_dic[c_position]))  # 这里面验证一下。现在是dict了，不是str。
        # print(count_dic[c_position])         # 每一列所对应的，所有元素的分别计数。

        score = 0             # 单列分数
        for ci in range(300):                # 根据我们之前生成的300对组合，一次看一下每一对在不在，在就计算。必须两个符号都在，才能叫在。
            # print(c[ci][0] and c[ci][1])    # 不能这样写，这样写就合并了，而不是且。必须像下面分开写。
            # 比如[('A', 'R'),的c[ci][0]=c[0][0]是A。count_dic[c_position]=count_dic[0]是字典{'-': 70, 'M': 1}的所有key'-''M'
            if c[ci][0] in count_dic[c_position] and c[ci][1] in count_dic[c_position]:   # 如果c这组元素都在字典里，说明可以计算了
                # print(str(c_position) + ':' + c[ci][0] + c[ci][1])   # 1549:RR。循环输出该position下所拥有的所有元素对，形如1549:RR 1549:R- 1549:--

                if c[ci][0] == c[ci][1] and count_dic[c_position][c[ci][0]] == 1:        # 元素相等且只有一个的时候不用算
                    pass
                elif c[ci][0] == c[ci][1] and count_dic[c_position][c[ci][0]] != 1:      # 元素相等且不唯一，n个中取两个
                    n = count_dic[c_position][c[ci][0]]             # 比如{'-': 70, 'M': 1}里的70
                    m = 2
                    # pair = math.factorial(n)/(math.factorial(m)*math.factorial(n-m))   # c(n,m) 计算对儿数。然后乘上该对的分数
                    pair = n * (n - 1) / 2                                               # m是2的时候可这样简化，不然阶乘很慢
                    if c[ci][0] == '-':
                        score += pair * matrix['*', '*']
                    else:
                        score += pair * matrix[c[ci][0], c[ci][1]]
                else:                                                                   # 元素不相等的情况下
                    pair = count_dic[c_position][c[ci][0]] * count_dic[c_position][c[ci][1]]         # 计算对儿数
                    if c[ci][0] == '-':
                        score += pair * matrix['*', c[ci][1]]
                    elif c[ci][1] == '-':
                        score += pair * matrix[c[ci][0], '*']
                    else:
                        score += pair * matrix[c[ci][0], c[ci][1]]
            else:
                pass

        # print('每列分数：'+str(score))   # 每列300对总共的分数
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

    c = combination_generator()

    # # 计算mafft的比对
    # for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列
    #     sp_li = ['sp']      # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的sp。并在开头事先存放一个字头。
    #     for i in range(1, 42):               # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'sp开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         sp_li.append(sp(align_file, c))
    #         print(time.ctime(), 'sp结束', i)
    #     with open(hmm_out_path + '/{ac}/get_sp'.format(ac=ac), 'w') as o:
    #         for every_sp in sp_li:
    #             o.write(str(every_sp) + '\n')
    #         o.close()

    # 计算随机100条小hmm指导的比对
    for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列
        sp_li = ['sp_align']      # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的sp。并在开头事先存放一个字头。
        for i in range(1, times+1):               # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
            print(time.ctime(), 'sp_align开始', i)
            align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_align.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
            sp_li.append(sp(align_file, c))
            print(time.ctime(), 'sp_align结束', i)
        with open(hmm_out_path + '/{ac}/get_sp_align'.format(ac=ac), 'w') as o:
            for every_sp in sp_li:
                o.write(str(every_sp) + '\n')
            o.close()
    # 计算随机100条小hmm指导的比对————只有match列版
    for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列
        sp_li = ['sp_align_m']      # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的sp。并在开头事先存放一个字头。
        for i in range(1, times+1):               # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
            print(time.ctime(), 'sp_align_m开始', i)
            align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_align.fam'.format(lp=local_path, s=i, ac=ac, pf=pf)
            sp_li.append(sp(align_file, c))
            print(time.ctime(), 'sp_align_m结束', i)
        with open(hmm_out_path + '/{ac}/get_sp_align_m'.format(ac=ac), 'w') as o:
            for every_sp in sp_li:
                o.write(str(every_sp) + '\n')
            o.close()

    # # 计算完整大数据集的hmm指导的比对
    # for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列
    #     sp_li = ['sp_align_big']      # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的sp。并在开头事先存放一个字头。
    #     for i in range(1, 42):               # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'sp_align_big开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_align_big.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         sp_li.append(sp(align_file, c))
    #         print(time.ctime(), 'sp_align_big结束', i)
    #     with open(hmm_out_path + '/{ac}/get_sp_align_big'.format(ac=ac), 'w') as o:
    #         for every_sp in sp_li:
    #             o.write(str(every_sp) + '\n')
    #         o.close()

    # # 计算我自己写的viterbi
    # for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列
    #     sp_li = ['sp_vi']      # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的sp。并在开头事先存放一个字头。
    #     for i in range(1, 42):               # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'sp_vi开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_vi.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         sp_li.append(sp(align_file, c))
    #         print(time.ctime(), 'sp_vi结束', i)
    #     with open(hmm_out_path + '/{ac}/get_sp_vi'.format(ac=ac), 'w') as o:
    #         for every_sp in sp_li:
    #             o.write(str(every_sp) + '\n')
    #         o.close()
    # # 计算我自己写的次优vi
    # for ac in onefamily_seqacs:          # 对于该家族中的所有的有结构序列
    #     sp_li = ['sp_subop_vi']      # 存放该家族中某条序列ac的，所有的阶梯px时，所对应的sp。并在开头事先存放一个字头。
    #     for i in range(1, 42):               # z4.2.0里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
    #         print(time.ctime(), 'sp_sub开始', i)
    #         align_file = '{lp}/repfam/{pf}/{ac}/{s}/random/{ac}_{pf}_subop_vi.fa'.format(lp=local_path, s=i, ac=ac, pf=pf)
    #         sp_li.append(sp(align_file, c))
    #         print(time.ctime(), 'sp_sub结束', i)
    #     with open(hmm_out_path + '/{ac}/get_sp_subop_vi'.format(ac=ac), 'w') as o:
    #         for every_sp in sp_li:
    #             o.write(str(every_sp) + '\n')
    #         o.close()