# 本程序在上一个程序执行完之后，需要提取每一个文件夹中的.hmm中的px数值，然后生成输出表等
# 针对大文件夹：1，2，3……

import os
import sys



def get_hmmpx(hmm_out_path, ac):

    with open(hmm_out_path + '/{ac}/{i}/{ac}_{pf}.hmm'.format(ac=ac, i=i, pf=pf)) as f:
        while 1:                                        # 遍历每一行
            line = f.readline()                         # 按行读取避免内存溢出
            if line[0:19] == 'STATS LOCAL FORWARD':     # 当你遇到这样的行时，证明你可以开始写了
                px = line.split()[3]             # 把px存在列表。然后就退出，完成任务。
                break
        f.close()

    return px



if __name__ == '__main__':
    # local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    local_path = '/data1/kzw'

    # pf = 'PF00083'
    pf = sys.argv[1]

    hmm_out_path = '{lp}/repfam/{pf}'.format(lp=local_path, pf=pf)  # 上一步的它们的结果文件保存的地址
    
    # onefamily_seqacs = ['A0A0H2VG78']  # 本次只会用到这一个,在这个目前的83家族中。后续可以通过for进行添加。
    seqac = sys.argv[2]
    onefamily_seqacs = [seqac]

    times = int(sys.argv[3])
    
    for ac in onefamily_seqacs:          # 对于该家族中的每一条序列
        px = ['hmmpx']  # 存放该家族中某条序列ac的，所有的阶梯px。并在开头事先存放一个字头。
        for i in range(1, times+1):  # z4_2里做了几个times/节点/阶梯就range几+1。从1文件夹开始。
            px.append(get_hmmpx(hmm_out_path, ac))      # 获取该序列的阶梯px数值并生成txt文件输出
        with open(hmm_out_path + '/{ac}/get_hmmpx'.format(ac=ac), 'w') as o:
            for every_px in px:
                o.write(str(every_px) + '\n')
            o.close()