# 本程序是我自己写的profile hmm从头构建多序列比对的程序。针对random文件夹。特别特别特别慢，我就是练练手，看看得了。
# 分为三部分：①初始参数确定 ②bw估计 ③viterbi回溯
# ①参数随机不太好，因此使mafft→hmmbuild现行估计了一下，以此作为初始参数。
import pickle
import time
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser



def aa_generator():    # 氨基酸字母生成器，按照hmmer顺序特供。

    aa_li = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    return aa_li



def get_char_allindex_string(char, string):   # 获取字符在某字符串中的所有索引

    all_id_li = [id for id in range(len(string)) if string.find(char, id) == id]

    return all_id_li



def phmm_parameter_initialization(local_path, pf, ac, i, aa_li):   # 本地地址，家族号，有结构序列号，第几次迭代文件夹，氨基酸生成器的结果
    # 对每一个迭代的random里面的hmm读取参数

    phmm_pi = {}    # 用于存放模型参数：M发射I发射MID转移矩阵
    L = 0           # 模型的长度

    # 我认为应该选择外层的hmm，而不是把random的有限序列重新做hmm。外层hmm才代表了本次迭代的整个数据集的整体情况，应以其为指导进行多序列比对。
    # 但实际来讲，后期这个模型长度太长了，几千几万的长度，算起来非常非常慢。
    # with open('{lp}/repfam/{pf}/{ac}/{i}/{ac}_{pf}.hmm'.format(lp=local_path, pf=pf, ac=ac, i=i)) as f:

    # 因此实际上，我们也许应该选择使用旧方案：把random的有限序列重新做hmm。以此来指导viterbi。
    # 需要z5_2_1_get_random.py启用备用搞事！！！！
    with open('{lp}/repfam/{pf}/{ac}/{i}/random/{ac}_{pf}_align_big.hmm'.format(lp=local_path, pf=pf, ac=ac, i=i)) as f:
    # with open('{lp}/repfam/{pf}/{ac}/{i}/random/{ac}_{pf}.hmm'.format(lp=local_path, pf=pf, ac=ac, i=i)) as f:

        while 1:
            l = f.readline()  # 我们让readline迭代器自己继续往下读行

            if l[0:2] == '//':  # 什么时候停止，//
                break

            if l[0:4] == 'LENG':  # 在长度行取模型长度
                L = int(l.split()[1])  # 模型长度，也就是M的数量。（不包括begin的数量）详情参见hmmer的hmm文件及其说明书

            if l[0:5] == 'HMM  ':   # 因为第一行也是hmm，所以这里加俩空格以作区分

                l = f.readline()    # 再往下读一行，跳过 m->m m->i m->d i->m i->i d->m d->d

                L1 = L + 1
                for t in range(L1):    # 接下来，就循环读取L+1次模型概率参数（包括了第一个0，也就是begin的数据）

                    l = f.readline()    # 每一次的第一行，是在此t时刻，M的发射概率
                    t_m_emiss = l[10:].strip().split()[0:20]    # 前10个字符是索引号，我们不要。从第11个开始，分割，且只取前20个。
                    for aa_i in range(20):
                        phmm_pi['{t}M{aa}'.format(t=t, aa=aa_li[aa_i])] = -float(t_m_emiss[aa_i])
                    phmm_pi['{t}MB'.format(t=t)] = 0    # 三种未知aa，概率0，对数负无穷。不设后面会报错。写0吧，不然整个都变成-inf
                    phmm_pi['{t}MZ'.format(t=t)] = 0    # 三种未知氨基酸。不过最好还是筛掉这种序列。
                    phmm_pi['{t}MX'.format(t=t)] = 0    # 三种未知氨基酸

                    l = f.readline()    # 每一次的第二行，是在此t时刻，I的发射概率
                    t_i_emiss = l[10:].strip().split()
                    for aa_i in range(20):
                        phmm_pi['{t}I{aa}'.format(t=t, aa=aa_li[aa_i])] = -float(t_i_emiss[aa_i])
                    phmm_pi['{t}IB'.format(t=t)] = 0    # 三种未知氨基酸
                    phmm_pi['{t}IZ'.format(t=t)] = 0    # 三种未知氨基酸
                    phmm_pi['{t}IX'.format(t=t)] = 0    # 三种未知氨基酸

                    l = f.readline()    # 每一次的第三行，是在此t时刻，MID的转换概率。其中，根据hmmer的习惯，我们不使ID互相转换。
                    t_trans = l[10:].strip().split()
                    phmm_pi['{t}M_M'.format(t=t)] = -float(t_trans[0])
                    phmm_pi['{t}M_I'.format(t=t)] = -float(t_trans[1])
                    if t_trans[2] == '*':
                        phmm_pi['{t}M_D'.format(t=t)] = np.NINF
                    else:
                        phmm_pi['{t}M_D'.format(t=t)] = -float(t_trans[2])
                    phmm_pi['{t}I_M'.format(t=t)] = -float(t_trans[3])
                    phmm_pi['{t}I_I'.format(t=t)] = -float(t_trans[4])
                    phmm_pi['{t}I_D'.format(t=t)] = np.NINF           # 延续hmmer等的习惯，不使ID之间有转移
                    if t_trans[5] == '0.00000':
                        phmm_pi['{t}D_M'.format(t=t)] = np.NINF
                    else:
                        phmm_pi['{t}D_M'.format(t=t)] = -float(t_trans[5])
                    phmm_pi['{t}D_I'.format(t=t)] = np.NINF           # 延续hmmer等的习惯，不使ID之间有转移
                    if t_trans[6] == '*':
                        phmm_pi['{t}D_D'.format(t=t)] = np.NINF
                    else:
                        phmm_pi['{t}D_D'.format(t=t)] = -float(t_trans[6])
        f.close()

    # 补充：在hmmer中，
    # 所有概率参数都存储为负自然对数概率，小数点右侧精度为五位数，四舍五入。例如，概率0.25被存储为−loge0.25=-ln0.25=1.38629。零概率的特殊情况存储为“*”。
    # 按照惯例，从不存在的删除状态0的不存在的转换被设置为log 1=0和log 0=−∞='*'。（不存在的状态就是0，0的转换就是*。）
    # 因此这里面，我们取负号的原因就是，我们希望它是ln(x)，而概率为零的ln就是np.NINF。

    return phmm_pi, L



def baum_welch(local_path, pf, ac, i,
               phmm_pi, L):     # 本地地址，家族号，有结构序列号，迭代文件夹。初始参数，模型长度。
    # 把fasta里的所有序列一条条的读取。直到所有序列都加进来。
    # 读取序列，每一条→自己的前向→自己的后向→用自己的前后向覆盖共用的的期望计数矩阵。直到所有序列都加进来而结束。
    # 最终的共用期望计数矩阵→最终的总期望概率矩阵。

    inpath = '{lp}/repfam/{pf}/{ac}/{i}/random/{ac}_{pf}_hmm_uni_seqs'.format(lp=local_path, pf=pf, ac=ac, i=i)  # 训比谁
    outpath = '{lp}/repfam/{pf}/{ac}/{i}/random/{ac}_{pf}'.format(lp=local_path, pf=pf, ac=ac, i=i)   # 新模参/比对输出

    # 该文件中所有序列的：

    # 存放总px
    px = 0

    # 减少运算次数
    L1 = L + 1

    # 总序列条数，暂时用不到
    # nseq = 0

    # 总的viterbi/次优viterbi导出的每条序列每个状态的内含氨基酸，以及每个状态应留出的最大长度
    align_vi = {}
    align_subop_vi = {}
    for l_k in range(L1):  # 先在vi回溯结果字典中，创建每个状态的key，并赋值0。后续不断以最大值赋给更新。
        align_vi[l_k] = 0  # 存储意义为 key状态：value该状态的最大长度
    for l_k in range(L1):  # 先在次优vi回溯结果字典中，创建每个状态的key，并赋值0。后续不断以最大值赋给更新。
        align_subop_vi[l_k] = 0  # 存储意义为 key状态：value该状态的最大长度


    with open(inpath) as f:  # 打开序列合集存在的文件

        for title, seq in SimpleFastaParser(f):       # 对于该合集中每一条序列

            # 总序列条数，暂时用不到
            # nseq += 1
            nj = len(seq)      # 该条序列长度
            nj1 = nj + 1       # 减少运算次数
            title_ac = title.split()[0]    # 当前序列的名字/登录号
            path = np.zeros((L1, nj1))     # viterbi回溯路径矩阵
            path_subop = np.zeros((L1, nj1))     # suboptimal viterbi回溯路径矩阵

            # Forward-------------------------------------------------------------------------------------------------

            # MID矩阵形状
            f_Mki = np.zeros((L1, nj1))     # Mki矩阵 匹配
            f_Iki = np.zeros((L1, nj1))     # Iki矩阵 插入
            f_Dki = np.zeros((L1, nj1))     # Dki矩阵 删除  # 这里给的参数指的是数量，而索引会比数量少一个。

            # 初始化：概率矩阵
            # f_Mki[0, 0] = 1      # 第0行，第0列的位置初始化为1，计算顺序具体见笔记

            # 初始化：将概率矩阵转化为对数概率矩阵
            f_Mki[0, 0] = 0
            f_Mki[0, 1:] = np.NINF
            f_Mki[1:, 0] = np.NINF
            f_Iki[:, 0] = np.NINF
            f_Dki[0, :] = np.NINF

            # 状态是0的时候，需要算I的第一行，也就是begin状态需要算。对于第0行每一个字符来说。
            # 注意这里aa，第1个字符对应的角标应该是seq0，以此类推，所以实际操作起来会比公式书上-1。
            for f_i in range(1, nj1):      # 对于I的第一行的每一列aa
                y = f_i - 1   # 矩阵索引，列
                e = phmm_pi['0I{aa}'.format(aa=seq[y])]         # emission，见笔记公式，单拎出来，为了减少重复计算，下同
                m = e + f_Mki[0, y] + phmm_pi['0M_I']           # 见公式fM，下同
                i = e + f_Iki[0, y] + phmm_pi['0I_I']           # 见公式fI，下同
                d = e + f_Dki[0, y] + phmm_pi['0D_I']           # 见公式fD，下同
                path[0, f_i] = np.argmax([m, i, d])                # viterbi索引。选mid中最大概率状态来源，记(用索引代其)在矩阵中
                path_subop[0, f_i] = np.random.choice([0, 1])      # 随机次优索引（注意ID之间不会有转移，下同）
                f_Iki[0, f_i] = np.logaddexp.reduce([m, i, d])     # 算前向（见公式，下同）

            # 从状态1开始到状态L、
            for f_k in range(1, L1):    # 从上往下，每到一个状态，先算M一行，再D一行，最后I一行。
                # 特别注意这里，是分开算的每一行，一行算完之后算另一个矩阵的另一行，不要像上面写的一个个算，上面的是错的。
                for f_i in range(1, nj1):                                           # 先算fm的一行
                    x = f_k - 1    # 矩阵索引，行。减少重复计算
                    y = f_i - 1    # 矩阵索引，列。减少重复计算
                    e = phmm_pi['{t}M{aa}'.format(t=f_k, aa=seq[y])]         # emission
                    m = e + f_Mki[x, y] + phmm_pi['{t}M_M'.format(t=x)]      # fm
                    i = e + f_Iki[x, y] + phmm_pi['{t}I_M'.format(t=x)]      # fi
                    d = e + f_Dki[x, y] + phmm_pi['{t}D_M'.format(t=x)]      # fd
                    path[f_k, f_i] = np.argmax([m, i, d])                    # viterbi矩阵
                    path_subop[f_k, f_i] = np.random.choice([0, 1, 2])       # subop_viterbi矩阵
                    f_Mki[f_k, f_i] = np.logaddexp.reduce([m, i, d])         # fm公式（所有的计算都是log之间的计算）
                for f_i in range(0, nj1):      # D比MI多最左边一列需要算，因此从零开始。  # 再算fd的一行
                    x = f_k - 1  # 矩阵索引，行。减少重复计算
                    m = f_Mki[x, f_i] + phmm_pi['{t}M_D'.format(t=x)]
                    i = f_Iki[x, f_i] + phmm_pi['{t}I_D'.format(t=x)]
                    d = f_Dki[x, f_i] + phmm_pi['{t}D_D'.format(t=x)]
                    path[f_k, f_i] = np.argmax([m, i, d])
                    path_subop[f_k, f_i] = np.random.choice([0, 2])
                    f_Dki[f_k, f_i] = np.logaddexp.reduce([m, i, d])
                for f_i in range(1, nj1):                                           # 最后算fi的一行
                    y = f_i - 1  # 矩阵索引，列。减少重复计算
                    e = phmm_pi['{t}I{aa}'.format(t=f_k, aa=seq[y])]
                    m = e + f_Mki[f_k, y] + phmm_pi['{t}M_I'.format(t=f_k)]
                    i = e + f_Iki[f_k, y] + phmm_pi['{t}I_I'.format(t=f_k)]
                    d = e + f_Dki[f_k, y] + phmm_pi['{t}D_I'.format(t=f_k)]
                    path[f_k, f_i] = np.argmax([m, i, d])
                    path_subop[f_k, f_i] = np.random.choice([0, 1])
                    f_Iki[f_k, f_i] = np.logaddexp.reduce([m, i, d])
            m = f_Mki[L, nj] + phmm_pi['{t}M_M'.format(t=L)]
            i = f_Iki[L, nj] + phmm_pi['{t}I_M'.format(t=L)]
            d = f_Dki[L, nj] + phmm_pi['{t}D_M'.format(t=L)]
            path_o = int(np.argmax([m, i, d]))                              # totheend的最大概率状态索引
            path_subop_o = [m, i, d].index(np.sort([m, i, d])[1])           # 次优随便选一个
            f_Mki_L1_nj1 = np.logaddexp.reduce([m, i, d])                   # 这就是该序列的px

            px += f_Mki_L1_nj1

            # print(f_Mki)
            # print(f_Iki)
            # print(f_Dki)
            # print(f_Mki_L1_nj1)   # 其实就是f_Mki[L+1, nj+1]的意思，但是它并不存在。因为矩阵没有那么大。

            # print(path)          # viterbi回溯路径矩阵
            # print(path_o)        # px处的viterbi来源/起点
            # print(path_subop)    # 次优viterbi回溯路径矩阵
            # print(path_subop_o)  # px处的次优viterbi来源/起点

            with open('{outpath}_px'.format(outpath=outpath), 'w') as pxfile:
                pxfile.write(str(px))
                pxfile.close()

            # # viterbi  -----------------------------------------------------------------------------------------------
            #
            # l_k = L
            # nj_i = nj
            # # for nj_i in range(nj, 0, -1):          # 从最后一个字符到最开始的第一个字符，对应了矩阵的nj到1。seq索引对应需要-1
            # while 1:
            #
            #     if l_k < 0:  # 字符从第nj个到第1个，小于一的全变成delete‘-’。状态从L到0，小于0证明结束了。
            #         break  # 回溯结束，跳出循环。
            #
            #     # 当矩阵xy索引为L，nj时，证明此时在右下角，右下角来源于矩阵外不存在的px，因此单独算。
            #     if nj_i == nj and l_k == L:          # 当前位置(也就是path矩阵中的位置)
            #         MID = 'MID'[path_o]           # 当前位置的最大概率状态。(如前所述矩阵和path_o存的都是索引012对应MID状态
            #         key = '{title}_{l_k}'.format(title=title_ac, l_k=l_k)    # 当前序列名_状态序号，作为次优比对结果字典的key
            #         if key not in align_vi:       # 如果这个 序列名_状态序号 key不存在，那就创建这个key
            #             align_vi[key] = []
            #         if MID == 'I':                      # 当前位置的状态是I的时候
            #             align_vi[key].append('MIs')    # 记录当前状态。(必是由当前状态序号M引领当前序号一系列I。注意这个倒着
            #             if nj_i - 1 >= 0:                    # 减1因为索引，最小0代表第一个字符。I状态<0啥也不用做。MD才需要插空
            #                 align_vi[key].append(str.lower(seq[nj_i - 1]))   # >=0代表字符还没结束，因此需要加入字符
            #             # 接下来，为下一个轮到的（其实是上一个，因为是回溯嘛）状态更新状态序号和字符索引。
            #             l_k = l_k                            # I状态的来源，状态序号不会变
            #             nj_i -= 1                            # 字符进入到下一个字符
            #         elif MID == 'M':                   # 当前位置的状态是M的时候
            #             align_vi[key].append('MMs')    # 记录当前状态。
            #             if nj_i - 1 < 0:                     # 字符索引<0证明字符已经被清空，因此插空
            #                 align_vi[key].append('-')
            #             else:
            #                 align_vi[key].append(seq[nj_i - 1])    # 否则插字符
            #             if len(align_vi[key]) > align_vi[l_k]:       # 若此序列当前状态长度>字典记录当前状态最大长度
            #                 align_vi[l_k] = len(align_vi[key]) - 1   # 则赋值。减1因为要减去最开头记录状态所占的位置
            #             # 接下来，为下一个轮到的（其实是上一个，因为是回溯嘛）状态更新状态序号和字符索引。
            #             l_k -= 1                              # M状态的来源，状态序号减一
            #             nj_i -= 1                             # 字符进入到下一个字符
            #         else:                              # 当前位置的状态是D的时候
            #             if MID_pre == 'I':                    # 若前面是I，这里变D，注意id之间应无转移。这里什么都不做，免得多写入
            #                 pass
            #             else:                                 # 其他情况正常写入
            #                 align_vi[key].append('DDs')  # 记录当前状态。
            #                 align_vi[key].append('-')    # 必定插空
            #             if len(align_vi[key]) > align_vi[l_k]:
            #                 align_vi[l_k] = len(align_vi[key]) - 1
            #             l_k -= 1
            #         MID_pre = MID
            #
            #     # 不在右下角的其他的情况。
            #     else:
            #         if l_k == L:                                      # 当x或者y到达0的时候，要分情况，把某些参数设到最大。
            #             MID = 'MID'[int(path[L, nj_i + 1])]           # 如若不然就会出错。出现参数溢出的情况。
            #         elif nj_i == nj:
            #             MID = 'MID'[int(path[l_k + 1, nj_i - 1])]
            #         elif nj_i < 0:
            #             MID = 'MID'[int(path[l_k + 1, 0])]
            #         else:
            #             MID = 'MID'[int(path[l_k + 1, nj_i + 1])]
            #         key = '{title}_{l_k}'.format(title=title_ac, l_k=l_k)
            #         if key not in align_vi:
            #             align_vi[key] = []
            #         if MID == 'I':
            #             if MID_pre != 'I' and nj_i - 1 >= 0:         # 前面不是I证明需要记录状态，是开头
            #                 align_vi[key].append('MIs')
            #             if nj_i - 1 >= 0:  # 小于零的时候就啥也不用做
            #                 align_vi[key].append(str.lower(seq[nj_i - 1]))
            #             l_k = l_k
            #             nj_i -= 1
            #         elif MID == 'M':
            #             if MID_pre != 'I':
            #                 align_vi[key].append('MMs')
            #             if nj_i - 1 < 0:
            #                 align_vi[key].append('-')
            #             else:
            #                 align_vi[key].append(seq[nj_i - 1])
            #             if len(align_vi[key]) > align_vi[l_k]:
            #                 align_vi[l_k] = len(align_vi[key]) - 1
            #             l_k -= 1
            #             nj_i -= 1
            #         else:
            #             if MID_pre == 'I':                    # 若前面是I，这里变D，注意id之间应无转移。这里什么都不做，免得多写入
            #                 pass
            #             else:                                 # 其他情况正常写入
            #                 align_vi[key].append('DDs')
            #                 align_vi[key].append('-')
            #             if len(align_vi[key]) > align_vi[l_k]:
            #                 align_vi[l_k] = len(align_vi[key]) - 1
            #             l_k -= 1
            #         MID_pre = MID
            # # print(align_vi)
            #
            #
            # # suboptimal vitrbi---------------------------------------------------------------------------------------
            #
            # l_k = L          # 状态
            # nj_i = nj        # 序列长度
            #
            # # for nj_i in range(nj, 0, -1):          # 从最后一个字符到最开始的第一个字符，对应了矩阵的nj到1。seq索引对应需要-1
            # while 1:
            #
            #     if l_k < 0:       # 字符从第nj个到第1个，小于一的全变成delete‘-’。状态从L到0，小于0证明结束了。
            #         break         # 回溯结束，跳出循环。
            #
            #     # 当矩阵xy索引为L，nj时，证明此时在右下角，右下角来源于矩阵外不存在的px，因此单独算。
            #     if nj_i == nj and l_k == L:          # 当前位置(也就是path矩阵中的位置)
            #         MID = 'MID'[path_subop_o]           # 当前位置的最大概率状态。(如前所述矩阵和path_o存的都是索引012对应MID状态
            #         key = '{title}_{l_k}'.format(title=title_ac, l_k=l_k)    # 当前序列名_状态序号，作为次优比对结果字典的key
            #         if key not in align_subop_vi:       # 如果这个 序列名_状态序号 key不存在，那就创建这个key
            #             align_subop_vi[key] = []
            #         if MID == 'I':                      # 当前位置的状态是I的时候
            #             align_subop_vi[key].append('MIs')    # 记录当前状态。(必是由当前状态序号M引领当前序号一系列I。注意这个倒着
            #             if nj_i - 1 >= 0:                    # 减1因为索引，最小0代表第一个字符。I状态<0啥也不用做。MD才需要插空
            #                 align_subop_vi[key].append(str.lower(seq[nj_i - 1]))   # >=0代表字符还没结束，因此需要加入字符
            #             # 接下来，为下一个轮到的（其实是上一个，因为是回溯嘛）状态更新状态序号和字符索引。
            #             l_k = l_k                            # I状态的来源，状态序号不会变
            #             nj_i -= 1                            # 字符进入到下一个字符
            #         elif MID == 'M':                   # 当前位置的状态是M的时候
            #             align_subop_vi[key].append('MMs')    # 记录当前状态。
            #             if nj_i - 1 < 0:                     # 字符索引<0证明字符已经被清空，因此插空
            #                 align_subop_vi[key].append('-')
            #             else:
            #                 align_subop_vi[key].append(seq[nj_i - 1])    # 否则插字符
            #             if len(align_subop_vi[key]) > align_subop_vi[l_k]:       # 若此序列当前状态长度>字典记录当前状态最大长度
            #                 align_subop_vi[l_k] = len(align_subop_vi[key]) - 1   # 则赋值。减1因为要减去最开头记录状态所占的位置
            #             # 接下来，为下一个轮到的（其实是上一个，因为是回溯嘛）状态更新状态序号和字符索引。
            #             l_k -= 1                              # M状态的来源，状态序号减一
            #             nj_i -= 1                             # 字符进入到下一个字符
            #         else:                              # 当前位置的状态是D的时候
            #             if MID_pre == 'I':                    # 若前面是I，这里变D，注意id之间应无转移。这里什么都不做，免得多写入
            #                 pass
            #             else:                                 # 其他情况正常写入
            #                 align_subop_vi[key].append('DDs')  # 记录当前状态。
            #                 align_subop_vi[key].append('-')    # 必定插空
            #             if len(align_subop_vi[key]) > align_subop_vi[l_k]:
            #                 align_subop_vi[l_k] = len(align_subop_vi[key]) - 1
            #             l_k -= 1
            #         MID_pre = MID
            #
            #     # 不在右下角的其他的情况。
            #     else:
            #         if l_k == L:
            #             MID = 'MID'[int(path_subop[L, nj_i + 1])]
            #         elif nj_i == nj:
            #             MID = 'MID'[int(path_subop[l_k + 1, nj_i - 1])]
            #         elif nj_i < 0:
            #             MID = 'MID'[int(path_subop[l_k + 1, 0])]
            #         else:
            #             MID = 'MID'[int(path_subop[l_k + 1, nj_i + 1])]
            #         key = '{title}_{l_k}'.format(title=title_ac, l_k=l_k)
            #         if key not in align_subop_vi:
            #             align_subop_vi[key] = []
            #         if MID == 'I':
            #             if MID_pre != 'I' and nj_i - 1 >= 0:         # 前面不是I证明需要记录状态，是开头
            #                 align_subop_vi[key].append('MIs')
            #             if nj_i - 1 >= 0:  # 小于零的时候就啥也不用做
            #                 align_subop_vi[key].append(str.lower(seq[nj_i - 1]))
            #             l_k = l_k
            #             nj_i -= 1
            #         elif MID == 'M':
            #             if MID_pre != 'I':
            #                 align_subop_vi[key].append('MMs')
            #             if nj_i - 1 < 0:
            #                 align_subop_vi[key].append('-')
            #             else:
            #                 align_subop_vi[key].append(seq[nj_i - 1])
            #             if len(align_subop_vi[key]) > align_subop_vi[l_k]:
            #                 align_subop_vi[l_k] = len(align_subop_vi[key]) - 1
            #             l_k -= 1
            #             nj_i -= 1
            #         else:
            #             if MID_pre == 'I':                    # 若前面是I，这里变D，注意id之间应无转移。这里什么都不做，免得多写入
            #                 pass
            #             else:                                 # 其他情况正常写入
            #                 align_subop_vi[key].append('DDs')
            #                 align_subop_vi[key].append('-')
            #             if len(align_subop_vi[key]) > align_subop_vi[l_k]:
            #                 align_subop_vi[l_k] = len(align_subop_vi[key]) - 1
            #             l_k -= 1
            #         MID_pre = MID
            # # print(align_subop_vi)
            # # print(title)

        f.close()

    # 把python的过程存储为pkl
    # with open('align_subop_vi.pkl', 'wb') as file:
    #     pickle.dump(align_subop_vi, file)
    # with open('align_subop_vi.pkl', 'rb') as filer:
    #     align_subop_vi = pickle.load(filer)
    # with open('align_subop_vi', 'w') as ooo:
    #     ooo.write(str(align_subop_vi))


    # with open(inpath) as f1:
    #
    #     # align_vi viterbi以对齐的格式完全写入
    #     with open('{outpath}_vi.fa'.format(outpath=outpath), 'w') as o1:
    #
    #         for title, seq in SimpleFastaParser(f1):  # 对每一条序列，以比对的格式重新写入
    #
    #             o1.write('>{title}'.format(title=title) + '\n')  # 先写title
    #             seq_vi = ''  # 用于存放该序列的以对齐的格式完全的比对
    #             for l_k in range(L1):  # 对于每一个状态
    #                 ac = title.split()[0]  # 该序列的名字
    #                 # 把字典中value倒着写，就是颠倒过来的正序序列，由大写字母带着一系列小写字母。每一个状态都有，把它取出来。
    #                 reverse_positive = align_vi['{ac}_{l_k}'.format(ac=ac, l_k=l_k)][-1:0:-1]  # 把字典中value倒着写
    #                 rp = ''
    #                 for char in reverse_positive:
    #                     rp += char  # 把颠倒正序的列表写入字符串
    #                 seq_vi += '{:.<{length}}'.format(rp, length=align_vi[l_k])  # 以该状态最大长度存在完全比对中
    #             o1.write(seq_vi + '\n')
    #
    #         o1.close()
    #
    #     f1.close()
    #
    #
    # with open(inpath) as f2:
    #
    #     # align_subop_vi 次优比对以对齐的格式完全写入
    #     with open('{outpath}_subop_vi.fa'.format(outpath=outpath), 'w') as o2:
    #
    #         for title, seq in SimpleFastaParser(f2):              # 对每一条序列，以比对的格式重新写入
    #
    #             o2.write('>{title}'.format(title=title) + '\n')   # 先写title
    #             seq_subop_vi = ''       # 用于存放该序列的以对齐的格式完全的比对
    #             for l_k in range(L1):   # 对于每一个状态
    #                 ac = title.split()[0]     # 该序列的名字
    #                 # 把字典中value倒着写，就是颠倒过来的正序序列，由大写字母带着一系列小写字母。每一个状态都有，把它取出来。
    #                 reverse_positive = align_subop_vi['{ac}_{l_k}'.format(ac=ac, l_k=l_k)][-1:0:-1]   # 把字典中value倒着写
    #                 rp = ''
    #                 for char in reverse_positive:
    #                     rp += char                        # 把颠倒正序的列表写入字符串
    #                 seq_subop_vi += '{:.<{length}}'.format(rp, length=align_subop_vi[l_k])    # 以该状态最大长度存在完全比对中
    #             o2.write(seq_subop_vi + '\n')
    #
    #         o2.close()
    #
    #     f2.close()


    return phmm_pi, px



if __name__ == '__main__':
    local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    hmm_out_path = '{lp}/repfam/PF00083'.format(lp=local_path)  # 上一步的它们的结果文件保存的地址

    pf = 'PF00083'
    onefamily_seqacs = ['O97467']  # 本次我只会用到这一个,在这个目前的83家族中，针对这一个做一下。后续可以通过for进行添加。

    for ac in onefamily_seqacs:  # 对于该家族中的所有的有结构序列

        px_li = ['px_li']

        # for i in range(1, 2):
        for i in range(1, 22):  # z4.2里做了几个times，结尾就range几+1。从1文件夹开始。
            print(time.ctime(), 'phmm开始', i)
            aa_li = aa_generator()    # 氨基酸顺序生成器
            phmm_pi, L = phmm_parameter_initialization(local_path, pf, ac, i, aa_li)    # 初始参数获得
            px = 0
            for bw_times_now in range(1):  # bw迭代更新1次
                phmm_pi, px = baum_welch(local_path, pf, ac, i, phmm_pi, L)     # 前向，后向，viterbi，次优，bw更新参数
            print(i, px)
            px_li.append(px)     # 写成with open
            print(time.ctime(), 'phmm结束', i)

        with open(hmm_out_path + '/{ac}/get_px_li_align_big'.format(ac=ac), 'w') as o:
            for every_px in px_li:
                o.write(str(every_px) + '\n')
            o.close()