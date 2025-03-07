# 本程序遍历pfama_full获取：一个pf号对应的所有full比对{PF00083_uniprot_seqac}序列登录号，目前存为txt就行


import os



def get_pfama_full_pf_seqac(pfama_full, full_seqac_path, pf):
    try:  # 创建一下，要是有就报错略过。
        os.makedirs('{}/repfam/{}/'.format(local_path, pf))            # 创建多级文件夹的命令。
    except:
        pass
    print('【起始】针对{}家族get_pfama_full_pf_seqac开始执行，正在读取Pfam-A.full'.format(pf))
    # 依然是逐行读取
    with open(pfama_full, 'r') as f, open(full_seqac_path, 'w') as o:

        nowline = 0

        while 1:     # （1）一直执行

            # # debug测试的时候还是正常写：
            # line = f.readline()
            # 跑的时候可以临时这样写：但是写代码的时候别这么写，你会不知道你程序中有哪些错误。
            try:                            # 由于某些行有未知的无关错误，导致程序中断，很烦，我们让它忽略错误继续执行
                line = f.readline()         # 你用一次就迭代一次，所以把它存为变量line，那line里面的就不会变了
                nowline += 1                # 读一行就记一次
                if nowline % 100000 == 0:
                    print('【过程】已读取{}行'.format(nowline))  # 可以做个可视化，每读取10万行就报告一次。
            except:
                print('【过程】警告：第{}行发生了一个异常如其所示：'.format(nowline), end='')
                print(line)

            # （2）一直执行，当碰到以下情况的时候，我们就进入我们的单独的小模块：我们仅对特定的所需家族才进行提取信息。
            if line[0:17] == '#=GF AC   {}'.format(pf):    # 0到16位，不到17。(小技巧：readline的[]字符索引超界不要怕，它不会报错，只会把所有的打出来)

                print('【过程】喜讯：我们找到了{}家族'.format(pf))

                seqnum = 0
                while 1:
                    l = f.readline()                   # 我们让readline迭代器自己继续往下读行

                    if l[0:4] == '#=GS':               # 遇到#=GS的时候就可以开始写入了
                        # print(l.strip().split(' '))    # ['#=GS', 'A0A0Q5YWZ0_9BURK/165-204', '', '', '', 'AC', 'A0A0Q5YWZ0.1']
                        l1 = l.strip().split(' ')[-1]  # 去掉收尾空格换行符号，并且用空格分割了这个字符串，并且只要最后一个东西。A0A0Q5YWZ0.1
                        l2 = l1.split('.')[0]          # 这最后一个部分再用.分割，取前面的部分A0A0Q5YWZ0
                        o.write(l2+'\n')
                        seqnum += 1

                    # 另外一个if是当符合要求的时候才写，不会有写多余的情况，所以这个终止判断放最后就行。
                    if l[0] != '#':                    # 什么时候停止呢，注意看文件的内容，前面全部都是带#的，后面是一些没有#的“序列号+序列”
                        o.write(str(seqnum))
                        break                          # 当遇到第一个不带#的地方，就证明我们取完了当前家族，后面的都不需要了。跳出这个写入的小模块。

                    # if not l :                         # 这个好像不需要这个，因为有//标识符
                    #     print('该文件读取完毕')
                    #     break

                print('【终止】您已经完成对{}家族的统计，共找到{}个full成员序列号。'.format(pf, seqnum))
                break  # （3）现在，我们该写的写完了，跳出整个while。直接到下面的把文件关闭的代码行。

        f.close()
        o.close()



if __name__ == '__main__':
    pfama_full = '/disk2/kzw/pfam/Pfam-A.full'             # 你的full全比对数据库唯一地址，确定了就无需再改
    pf = 'PF00083'                                         # 当前输出的是哪一个pf家族，可以更改，而且变成for循环多个家族
    local_path = '/disk2/kzw'                              # 我的路径
    full_seqac_path = '{}/repfam/{}/{}_full_seqac'.format(local_path, pf, pf)  # 输出地址，full全比对序列号文件，改成你的

    get_pfama_full_pf_seqac(pfama_full, full_seqac_path, pf)