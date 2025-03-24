# 本程序使用shell执行：提取单条序列，blast命中，clustalo命中+mat+dnd，调用cluster_dnd_nclus生成clu。
# 本程序是自动化的，但中间某次出于未知原因，可能是任务之间不同步，因此必须手动注释然后一个一个流程完成。

import os                      # 在python中执行shell的一种方法
# import cluster_dnd_nclus       # 调用一个我自己写的py



def ana_seq_matdnd_sh(stseqac_path, db, out_path):

    # （一）先创建blast的数据库
    # os.system('makeblastdb -in {lp}/repfam/{pf}/{pf}_uniprot_seq/{pf}_uniprot_seqs.fasta \
    #                     -dbtype prot -parse_seqids -title "{pf}_uniprot_seqs" \
    #                     -out {db}'.format(lp=local_path, pf=pf, db=db))

    seqsli = []
    with open(stseqac_path) as f:
        for i in f.readlines()[0:-1]:
            seqsli.append(i.strip())
        f.close()
        print(seqsli)
    li1 = []

    for i in seqsli:

        try:  # 创建一下，要是有就报错略过。
            os.makedirs(out_path+'/{}/'.format(i))  # 创建多级文件夹的命令。
        except:
            pass

        # # 请按照步骤依次手动解除注释、执行，一起执行好像有点反应不过来？
        #（一）在db里根据序列号i去entry搜寻目标序列，输出为一个单条序列的fasta文件。
        # os.system('blastdbcmd -db {db} -entry {i} -out {o}/{i}/{i}.fasta'.format(db=db, o=out_path, i=i))
        # 在db中搜寻跟此序列相像的序列，以11格式输出搜索结果。同时限制evalue的值，并且留出10000的搜索结果余量。
        # os.system('blastp -db {db} -query {o}/{i}/{i}.fasta -outfmt 11 -out {o2}/{i}/{i}_{pf}_bp.asn \
        #             -evalue 9.9e-150 -max_target_seqs 10000 -num_threads 4'\
        #             .format(db=db, o=out_path, i=i, o2=out_path, pf=pf))
        #（二）将11号万用存储格式中的结果，再提取为7格式的out文件。# 备用：-outfmt "7 sacc evalue bitscore score" (pident)
        # os.system('blast_formatter -archive {o}/{i}/{i}_{pf}_bp.asn \
        #           -outfmt "7 sacc" -out {o}/{i}/{i}_{pf}_bp.out'\
        #           .format(o=out_path, i=i, pf=pf))

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 如果你想批量查看bp.out文件，统计每一个序列搜到多少 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        #（伪三）打开每一个out文件，把序列号和其对应的搜到的序列数量，存在li1里。
        with open('{o}/{i}/{i}_{pf}_bp.out'.format(o=out_path, i=i, pf=pf), 'r') as fo:
            line5 = fo.readlines()[4].split(' ')[1]
            li1.append(i + ' ' + line5)
            fo.close()
        print(li1)

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 如果你想对搜出来的bp.out文件进行一个clustalo分析 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        #（三）根据out文件搜到的上千条序列号的列表，去提取序列们，合成一个fasta合集。
        # 这一步有问题'Error: [blastdbcmd] Skipped #'是正常的，文件里确实有井号，文件结构问题。第（二）步必须是：-outfmt "7 sacc"
        # os.system('blastdbcmd -db {db} -entry_batch {o}/{i}/{i}_{pf}_bp.out>{o}/{i}/{i}_{pf}_bpseqs.fasta'\
        #           .format(db=db, o=out_path, i=i, pf=pf))
        #（四）用clustalo去分析这上千条序列。注意这里的clustalo怎么也找不到，path配置完了也这样，那么就用绝对路径吧。
        # os.system('{lp}/clustalo/clustalo -i {o}/{i}/{i}_{pf}_bpseqs.fasta -o {o}/{i}/{i}_{pf}_bpseqs.sto --outfmt=st --force \
        #           --distmat-out={o}/{i}/{i}_{pf}_bpseqs.mat --guidetree-out={o}/{i}/{i}_{pf}_bpseqs.dnd --full'\
        #           .format(lp=local_path, o=out_path, i=i, pf=pf))
        #（五）调用了一下自己写的小玩意，转换一下dnd
        # cluster_dnd_nclus.cluster_dnd_nclus('{o}/{i}/{i}_{pf}_bpseqs.dnd'.format(o=out_path, i=i, pf=pf),\
        #                                     '{o}/{i}/{i}_{pf}_bpseqs_dnd.cluster'.format(o=out_path, i=i, pf=pf))
        print('mmm')

    print('我喵喵喵')

if __name__ == '__main__':
    # local_path = '/disk2/kzw'  # 我的路径
    local_path = '/home/handsomekk'  # 我的30pc
    pf = 'PF00083'  # 你的家族号
    stseqac_path = '{}/repfam/{}/{}_stseqac'.format(local_path, pf, pf)  # 有结构序列的序列号 文件
    db = '{lp}/repfam/{pf}/{pf}_uniprot_seq/db/{pf}_uniprot_seqs'.format(lp=local_path, pf=pf)
    out_path = '{lp}/repfam/{pf}/{pf}_stseq_ana'.format(lp=local_path, pf=pf)
    ana_seq_matdnd_sh(stseqac_path, db, out_path)