# 本程序遍历pdbmap获取：pf号对应的结构号和{PF00083_uniprot_seqac}序列登录号，存为txt


import os



def get_pdbmap_pf_stseqac(pdbmap, out_path, pf):

    try:  # 创建一下，要是有就报错略过。
        os.makedirs('{}/repfam/{}/'.format(local_path, pf))            # 创建多级文件夹的命令。
    except:
        pass
    print('【起始】针对{}家族get_pdbmap_pf_stseqac开始执行，正在读取pdbmap'.format(pf))

    # 依然是逐行读取
    with open(pdbmap, 'r') as f, open(out_path.format(pf), 'w') as o:

        nowline = 0

        while 1:   # （1）一直执行

            # # debug测试：
            # line = f.readline()
            try:                         # 由于某些行有未知的无关错误，导致程序中断，忽略错误继续执行
                line = f.readline()      # 用一次就迭代一次，把它存为变量line，line里面的不会变
                nowline += 1             # 读一行就记一次
                if nowline % 100000 == 0:
                    print('【过程】已读取{}行'.format(nowline))   # 可以做个可视化，每读取10万行就报告一次。
            except:
                print('【过程】警告：第{}行发生了一个异常如其所示：'.format(nowline), end='')
                print(line)
            if not line:                     # 中途发现总有报错，结果发现是因为有的家族在这里没有结构，所以加一个终止判定。
                print('【终止】您已经完成对pdbmap的全部检索！没有找到{}家族！'.format(pf))
                break

            li = line.strip().split(';\t')              # ['7e2y', 'R', '53-400', '7tm_1', 'PF00001', 'P08908', '53-400;']
            # （2）一直执行，当碰到以下情况的时候，我们就进入我们的单独的小模块：我们仅对特定的所需家族才进行提取信息。
            if li[4] == pf:                             # 第五位取出来，是我们需要的家族的时候，就可以开始了。这是第一行。

                print('【过程】喜讯：我们找到了{}家族'.format(pf))
                s = set()                             # 我们创建一个set，用于存放所有有结构的序列的登录号，且自动去重
                s.add(li[5])                          # 第一行处理完毕

                while 1:
                    l = f.readline()                  # 第一行在外面已经处理完了，从这里开始是第二行的，继续往下读啦~

                    if not l:                         # 由于这个文件没有终止标识符，需要首先判断是不是到了文件结尾，结尾就不读了
                        break

                    li2 = l.strip().split(';\t')      # 没到结尾就取第五位看看是不是我们需要的
                    if li2[4] != pf:                  # 不是需要的就终止。因为这次只取一个家族，不是就说明我们要的家族在上一行已经取完了。
                        break                         # 因为pdbmap是按家族排列的，每一个家族都是排在一起的

                    s.add(li2[5])                     # 没终止，是我们需要的，那就请把第六位序列号添加到set集合

                for i in s:                    # 最后把set中的元素挨个写进文件o
                    o.write(i+'\n')

                seqnum = len(s)
                o.write(str(seqnum))           # 一共多少条有结构序列也写一下

                print('【终止】您已完成对{}家族的统计，共找到{}个有结构序列号。'.format(pf, seqnum))

                break  # （3）现在，我们该写的写完了，跳出整个while。直接到下面的把文件关闭的代码行。

        f.close()
        o.close()



if __name__ == '__main__':
    pdbmap = '/disk2/kzw/pfam/pdbmap'           # 你的pdbmap唯一地址，确定了就无需再改
    pf = 'PF00083'                              # 当前输出的是哪一个pf家族，可以更改，甚至变成for循环多个家族
    local_path = '/disk2/kzw'                   # 我的路径
    stseqac_path = '{}/repfam/{}/{}_stseqac'.format(local_path, pf, pf)  # 输出地址，有结构序列号文件。改成你的

    get_pdbmap_pf_stseqac(pdbmap, stseqac_path, pf)