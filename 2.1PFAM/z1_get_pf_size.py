# 统计所有类型的pf号以及其对应的成员数目
# 本程序获取：PF号+类型+家族数量，先是存在了TXT里，然后会转换成csv格式

import csv



# 一次性读取可能内存溢出
# with open("C:/Users/ts/Desktop/t.txt", 'r') as f:
#     content = f.read()
#     print(content)

# 逐行读取
def get_pf_size(local_path, out_path, file):
    print('【起始】针对pfam全局进行统计，get_pf_size开始执行，正在读取{}'.format(file))
    with open(local_path+file, 'r') as f, open(out_path+'/{}.txt'.format(file), 'w') as o:

        count = 0

        while 1:
            # for line in f.readlines():
            try:
                line = f.readline()    # 有一行似乎有点问题，无法正常读，用try跳过
            except:
                pass
            # print(line)
            if line[0:7] == '#=GF AC':
                o.write(line[10:-1] + ',')   # 第十位到倒数第一位，并且以','结尾。
            if line[0:7] == '#=GF TP':
                o.write(line[10:-1] + ',')
            if line[0:7] == '#=GF SQ':
                o.write(line[10:])          # 不截至到倒数第一位，原字符串就带个回车。

            count += 1
            if count % 100000 == 0:
                print('【过程】已读取{}行'.format(count))

            if not line:
                break

        f.close()
        o.close()

    # newline='' 就没有多余的空白行了
    with open(out_path+'/{}.txt'.format(file), 'r') as f, open(out_path+'/{}.csv'.format(file), 'w', newline='') as o:

        writer = csv.writer(o)              # 这里是o，不是文件名

        while 1:
            # for line in f.readlines():
            line = f.readline()
            data = line.strip().split(',')  # 去除空格和回车，并且按照‘,’分割成列表
            writer.writerow(data)

            if not line:
                break

        f.close()
        o.close()

    print('【终止】针对pfam全局的统计已完成！根据{}生成两份文件。'.format(file))



if __name__ == '__main__':
    # 你可以选择统计哪一个文件：file1 or 2
    # 其他是固定格式，只需要将local_path改成你自己的数据库所在的path，out_path改成你自己的目录
    # 注意：有一个家族的数量统计是空白，因此造成列表篡位，记得手动调整一下
    # 其实我也不确定我具体用的是哪一版本的pfam数据了，大概是在那个附近吧
    local_path = '/disk2/kzw'                           # 71
    out_path = '/disk2/kzw/myproject/outfile'           # 输出地址
    file1 = 'Pfam-A.full'                               # full比对
    file2 = 'Pfam-A.full.uniprot'                       # uniprot比对
    get_pf_size(local_path+'/pfam/', out_path, file1)
    get_pf_size(local_path+'/pfam/', out_path, file2)