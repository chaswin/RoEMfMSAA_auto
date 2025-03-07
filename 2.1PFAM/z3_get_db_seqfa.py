# 由于网页下载慢，且有的序列更新迭代快已经无法下载、不太稳定，get_url_seqfa已经弃用。
# 本程序遍历pfamseq或者uniprot文件：根据序列号列表去获取我们要的序列。我们会把所有的序列放到一个大fasta里，我们认为这样更利于后续分析处理。

# 自己原先思考的两个方案：①多次遍历：每次遍历查找一个序列。②单次遍历：一次性提取所有，每一行都判断一次。
# 但是现在我们有了biopython这个强大的工具，请先去学习biopython的3、4、5章，最后看一眼22.1.1小节，你的问题会被很短的几行代码解决。
# 其实主要是第5章，就已经显示了两个可用的迭代器。22章三类似于综合应用。在第5章中挑选合适的迭代器，然后像22那样应用就可以了。

import os
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio import SwissProt
from tqdm import tqdm



def load_path(local_path, pf):   # 加载所需path，并返回这些path。同时创建本不存在的文件夹
    # 序列号文件
    stseqac_path = '{}/repfam/{}/{}_stseqac'.format(local_path, pf, pf)                # 有结构序列的序列号 文件
    full_seqac_path = '{}/repfam/{}/{}_full_seqac'.format(local_path, pf, pf)          # full全比对序列的序列号 文件
    uniprot_seqac_path = '{}/repfam/{}/{}_uniprot_seqac'.format(local_path, pf, pf)    # uniprot全比对序列的序列号 文件
    # 提取出来的序列要存放的地方
    stseq_path = '{}/repfam/{}/{}_stseq/'.format(local_path, pf, pf)                   # 有结构的序列，存放地址 文件夹
    full_seq_path = '{}/repfam/{}/{}_full_seq/'.format(local_path, pf, pf)             # full全比对序列，存放地址 文件夹
    uniprot_seq_path = '{}/repfam/{}/{}_uniprot_seq/'.format(local_path, pf, pf)       # uniprot全比对序列，存放地址 文件夹

    try:  # 创建一下，要是有就报错略过。
        os.makedirs(stseq_path)            # 创建多级文件夹的命令。
    except:
        pass
    try:  # 之前写在一起了，结果第一个有，第二个没有，第二个就不创建了，那就单独写一下。
        os.makedirs(full_seq_path)
    except:
        pass
    try:
        os.makedirs(uniprot_seq_path)
    except:
        pass
    #      0,1,2                                              3,4,5
    return stseqac_path, full_seqac_path, uniprot_seqac_path, stseq_path, full_seq_path, uniprot_seq_path


def get_seqac_set(seqacfile_path):        # 我们把seqac文件中的ac全都存到一个set中
    seqacfile_name = seqacfile_path.split('/')[-1]   # 给地址切个片，取文件名
    print('【起始】开始处理{}文件，制作set集合'.format(seqacfile_name))
    seqac_set = set()
    with open(seqacfile_path) as f:               # 打开seqac的文件
        for line in f.readlines()[0:-1]:          # 每一行都是一个序列号。(除了最后一行，是我自己记录的序列号数量，具体参见get_seqac.py系列文件)
            seqac_set.add(line.strip())           # 把序列号加入集合set
        f.close()
    with open(seqacfile_path) as f:               # 为什么要再打开一遍？readline属于迭代器，用一次少一次，自动向下读，用完就没有了。必须重新打开一遍才有。
        should_num = f.readlines()[-1].strip()    # 把最后一行我自己记录的序列号数量保存在这个变量中。
    print("【终止】{}文件包含序列号{}个，实际非冗余序列{}个。".format(seqacfile_name, should_num, len(seqac_set)))
    return seqac_set                      # 该函数返回的就是序列号的集合set


# 参数：数据库地址，三个输出文件地址，pf号，三种输出文件命名格式，三个序列号合集
def from_fa_extract_seqs(db,
                         out_path1, out_path2, out_path3,
                         pf,
                         type1, type2, type3,
                         seqac_set1, seqac_set2, seqac_set3):   # 根据seqac_set集合，从pfamseq/uniprot(fasta)数据库中提取序列们。
    print('【起始】开始根据{}的三个集合，在{}中提取序列'.format(pf, db))
    numst = 0
    numfull = 0
    numuni = 0
    with open(db) as f, open(out_path1+'{}_{}.fasta'.format(pf, type1), 'w') as o1, \
                        open(out_path2+'{}_{}.fasta'.format(pf, type2), 'w') as o2, \
                        open(out_path3+'{}_{}.fasta'.format(pf, type3), 'w') as o3:   # 打开数据库，同时创建三个要写入序列的fasta
        for title, seq in tqdm(SimpleFastaParser(f)):                 # 对于该数据库中的每项中的，标题和序列。注意，title不包含'>'。
            ac = title.split('.', 1)[0]              # 根据具体情况进行修改
            # 注意，这里犯了一个严重的id号不一致导致的后续blast数据库query序列的时候找不到的情况。
            # 之前写入下面三个文件的时候，写的是>V6AU82.1这样的带版本号的id，但seqac文件都是不带版本号的。
            # 导致无法在sh脚本快速的使用seqac列表，for循环依次blastp -query序列号 查询。
            # 于是我们这里操作一下，去掉那个版本号。统一规定：我们使用的fasta里面不允许带版本号，要纯净id。
            if ac in seqac_set1:                  # 如果标题中序列号在我们要的st集合中
                o1.write('>{}\n{}\n'.format(ac + ' ' + title.split(' ',1)[1], seq))              # 就写入o1
                numst += 1                                            # 写一条 记一条
            if ac in seqac_set2:                  # 如果标题中序列号在我们要的st集合中
                o2.write('>{}\n{}\n'.format(ac + ' ' + title.split(' ',1)[1], seq))              # 就写入o2
                numfull += 1                                          # 写一条 记一条
            if ac in seqac_set3:                  # 如果标题中序列号在我们要的st集合中
                o3.write('>{}\n{}\n'.format(ac + ' ' + title.split(' ',1)[1], seq))              # 就写入o3
                numuni += 1                                           # 写一条 记一条
        f.close()
        o1.close()
        o2.close()
        o3.close()
    print("【终止】{}条序列成功写入{}_{}.fasta文件中。".format(numst, pf, type1))
    if numst < len(seqac_set1):                                          # 要是实际写入的比集合中的序列少
        print('【终止】警告：{}集合中{}个序列号没有被找到。'.format('stseqac_set', len(seqac_set1)-numst))
    print("【终止】{}条序列成功写入{}_{}.fasta文件中。".format(numfull, pf, type2))
    if numfull < len(seqac_set2):                                        # 要是实际写入的比集合中的序列少
        print('【终止】警告：{}集合中{}个序列号没有被找到。'.format('full_seqac_set', len(seqac_set2)-numfull))
    print("【终止】{}条序列成功写入{}_{}.fasta文件中。".format(numuni, pf, type3))
    if numuni < len(seqac_set3):                                         # 要是实际写入的比集合中的序列少
        print('【终止】警告：{}集合中{}个序列号没有被找到。'.format('uniprot_seqac_set', len(seqac_set3)-numuni))


# # 之前预判错误，以为st和uni的序列在dat文件里。其实在uniprot(fasta)文件里。因此这个函数弃用。
# # 参数：数据库地址，有结构序列输出地址，uni全比对序列输出地址，pf号，结构序列输出命名格式，uni序列输出命令格式，结构序列号合集，uni序列号合集
# # 根据seqac文件，从形如xxx.dat的文件中提取序列们。
# def from_dat_extract_seqs(db, st_out_path, uni_out_path, pf, sttype, unitype, stseqac_set, uniprot_seqac_set):
#     f = open(db)
#     records = SwissProt.parse(f)          # 打开这个数据库文件，并用biopython提供的工具解析
#     num1 = 0
#     num2 = 0                              # 因为stseq和uniseq都在这个db，可以同时写两份文件
#     with open(st_out_path+'{}_{}.fasta'.format(pf, sttype), 'w')as o1, open(uni_out_path+'{}_{}.fasta'.format(pf, unitype), 'w')as o2:
#         for i in records:
#             if i.accessions[0] in stseqac_set:
#                 ac = i.accessions[0]                             # 序列号，又叫登录号
#                 sn = i.entry_name                                # 短名
#                 fn = i.description.split('=')[-1].split(';')[0]  # 全名。等号后面的，分号之前的
#                 eco = i.comments[0].split(' ')[-1].split('.')[0]  # eco号。空格分开的最后面的，‘.’之前的
#                 seq = i.sequence                                 # 序列
#                 o1.write('>{} {} {} {}\n{}\n'.format(ac, sn, fn, eco, seq))   # 完全按照标准fasta格式写入
#                 num1 += 1
#             if i.accessions[0] in uniprot_seqac_set:
#                 ac = i.accessions[0]
#                 sn = i.entry_name
#                 fn = i.description.split('=')[-1].split(';')[0]  # 等号后面的，分号之前的
#                 eco = i.comments[0].split(' ')[-1].split('.')[0]  # 空格分开的最后面的，‘.’之前的
#                 seq = i.sequence
#                 o2.write('>{} {} {} {}\n{}\n'.format(ac, sn, fn, eco, seq))
#                 num2 += 1
#         o1.close()
#         o2.close()
#     print("写入：{}条序列成功写入{}_{}.fasta文件中。".format(num1, pf, sttype))
#     if num1 < len(stseqac_set):  # 要是实际写入的比集合中的序列少
#         print('警告：{}_stseqac中{}个序列号没有被找到。'.format(pf, len(stseqac_set)-num1))
#     print("写入：{}条序列成功写入{}_{}.fasta文件中。".format(num2, pf, unitype))
#     if num2 < len(uniprot_seqac_set):  # 要是实际写入的比集合中的序列少
#         print('警告：{}_uniprot_seqac中{}个序列号没有被找到。'.format(pf, len(uniprot_seqac_set)-num2))



if __name__ == '__main__':
    # 使用说明：你只需要在main中确定你pfam_database，uni_database，local_path，pf即可。其中pf号是可变的，可改成for循环对更多家族批量执行。

    pfam_database = "/disk2/kzw/pfam/pfamseq"  # 你的pfamseq唯一地址，确定了就无需再改。文件非常大，该文件仅包含full比对的序列。
    uni_database = '/disk2/kzw/pfam/uniprot'   # 你的uniprot唯一地址，确定了就无需再改。文件非常大。该文件包含(st)uniprot比对的序列。
    local_path = '/disk2/kzw'                  # 我的路径
    pf = 'PF00083'                             # 你的家族号

    # dat_database = '...../xxx.dat'             # 还有一种数据库，称为swiss的dat格式，当你需要从中提取信息可以使用from_dat_extract_seqs函数
    # pfam_database = "/home/handsomekk/pycharmproject/pythonproject/pfamseq"  # 测试时，模仿pfamseq用的假数据库
    # local_path = '/home/handsomekk'            # 个人电脑的路径，测试用

    # type是人为定义命名时候的格式，如所示：type = 'stseqs' 'full_seqs' 'uniprot_seqs'

    paths = load_path(local_path, pf)

    stseqac_set = get_seqac_set(paths[0])            # 有结构的序列号的集合
    full_seqac_set = get_seqac_set(paths[1])         # full比对的序列号的集合
    uniprot_seqac_set = get_seqac_set(paths[2])      # uniprot比对序列号的集合

    # 新版，只需要遍历一次，同时写三个文件。 # uni数据库，三个文件的输出地址，pf号，三个输出文件的命名格式，三个序列号集合。
    from_fa_extract_seqs(uni_database,
                         paths[3], paths[4], paths[5],
                         pf,
                         'stseqs', 'full_seqs', 'uniprot_seqs',
                         stseqac_set, full_seqac_set, uniprot_seqac_set)


    # # 旧版需要遍历三次，等不了了
    # from_fa_extract_seqs(uni_database, paths[3], pf, 'stseqs', stseqac_set)
    # from_fa_extract_seqs(pfam_database, paths[4], pf, 'full_seqs', full_seqac_set)
    # from_fa_extract_seqs(uni_database, paths[5], pf, 'uniprot_seqs', uniprot_seqac_set)
    # # 停用函数
    # from_dat_extract_seqs(uni_database, paths[3], paths[5], pf, 'stseqs', 'uniprot_seqs', stseqac_set, uniprot_seqac_set)

    # # 测试而已
    # set1 = get_seqac_set('/home/handsomekk/repfam/PF00083/PF00083_stseqac')
    # set2 = get_seqac_set('/home/handsomekk/repfam/PF00083/PF00083_full_seqac')
    # # set2 = get_seqac_set('./PF00083_full_seqac')
    # set3 = get_seqac_set('/home/handsomekk/repfam/PF00083/PF00083_uniprot_seqac')