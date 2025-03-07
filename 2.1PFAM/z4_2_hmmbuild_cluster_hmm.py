# 本程序使用shell执行【首先】对上一步bp得到的71条序列用mafft比对生成fa格式，hmmbuild建模，hmmsearch得out；提136seqac，生成seqs.fasta
#【然后】重复：(fa格式的替代品)sto格式，hmmbuild，hmmsearch得到out，提取序列号，根据序列号生fasta文件。

import os
import sys



def nitial_attempts(db, hmm_db, seqana_bpseqs_path,
                    hmm_out_path,
                    initial_evalue,
                    onefamily_seqacs):   # 【首先】初始的一次过程。

    for ac in onefamily_seqacs:  # 对于该家族的每一个序列号，都做这样的事情：

        try:  # 创建一下，要是有就报错略过。
            os.makedirs(hmm_out_path + '/{ac}/0/'.format(ac=ac))  # 创建多级文件夹的命令。在该家族的文件夹下，为每个序列号都单独建一个文件夹。
        except:
            pass

        # 一、这是最初最初的时候，利用blastp给出来的那个搜索出来的71条序列的合集。

        # 1.用blastp生成的序列合集fasta作为前体
        print('【开始】mafft多序列比对fasta'.format(ac=ac))
        os.system('mafft --retree 1 --maxiterate 0 --nofft --parttree \
                  --quiet \
                  {inpath}/{ac}/{ac}_{pf}_bpseqs.fasta > {outpath}/{ac}/0/{ac}_{pf}.fa'
                  .format(inpath=seqana_bpseqs_path, ac=ac, outpath=hmm_out_path, pf=pf))    # 用mafft生成的以fa格式存储的序列比对
        print('mafft比对结束，得到.fa'.format(ac=ac))

        # 2.以刚得到的序列比对结果fa为前体
        print('【开始】hmmbuild以.fa文件构建隐马模型')
        # 注意写你自己的hmmbuild地址，比如{lp}/hmmer34/bin/hmmbuild，当然如果你path配置好了也可以直接写‘hmmbuild’。下同理
        # 这里面，-enone命令是不进行有效序列的选取，但这样似乎会造成过拟合，所以弃用
        # 此外，可以在结尾加上 > /dev/null，以“删除”所有的屏幕输出
        os.system('hmmbuild --cpu 30 {outpath}/{ac}/0/{ac}_{pf}.hmm {inpath}/{ac}/0/{ac}_{pf}.fa'
                  .format(outpath=hmm_out_path, inpath=hmm_out_path, ac=ac, pf=pf))             # 用该序列比对结果去生成hmm
        print('hmmbuild建模结束，得到.hmm'.format(ac=ac))

        # 3.以生成的hmm作为前体
        print('【开始】hmmsearch搜索')
        # 注意写你自己的hmmsearch地址，也可以直接‘hmmsearch’，下同理
        os.system('hmmsearch --noali --notextw --cpu 30 \
                  -E {ie} \
                  -o {outpath}/{ac}/0/{ac}_{pf}_hmm_uni.out \
                  {inpath}/{ac}/0/{ac}_{pf}.hmm \
                  {db}'
                  .format(outpath=hmm_out_path, inpath=hmm_out_path, ac=ac, db=hmm_db, ie=initial_evalue, pf=pf))  # 用hmmsearch搜出out
        print('hmmsearch搜索结束，得到.out'.format(ac=ac))

        # hmmsearch的一般用法
        #（1）
        # hmmsearch '/home/handsomekk/repfam/PF00083/O97467/O97467_PF00083_bpseqs.hmm'
        # '/home/handsomekk/repfam/PF00083/PF00083_uniprot_seq/PF00083_uniprot_seqs.fasta' >
        # /home/handsomekk/repfam/PF00083/O97467/O97467_PF00083_bpseqs_hmm_uni.out
        #（2）
        # hmmsearch --noali --notextw -E 1e-80
        # --incE 1e-80  # 没必要
        # -o /home/handsomekk/repfam/PF00083/O97467/O97467_PF00083_bpseqs_hmm_uni_2.out
        # --tblout O97467_PF00083_bpseqs_hmm_uni_2.xls
        # '/home/handsomekk/repfam/PF00083/O97467/O97467_PF00083_bpseqs.hmm'
        # '/home/handsomekk/repfam/PF00083/PF00083_uniprot_seq/PF00083_uniprot_seqs.fasta'

        # 4.1把out中的序列号的部分提取出来。
        count = 0  # 计算序列号的数量
        hmm_seqac = []   # 存放序列号的列表
        with open('{path}/{ac}/0/{ac}_{pf}_hmm_uni.out'.format(path=hmm_out_path, ac=ac, pf=pf)) as f:  # 打开out
            while 1:                                        # 遍历每一行
                line = f.readline()                             # 按行读取避免内存溢出
                if line[0:11] == '    -------':                 # 当你遇到这样的行时，证明你可以开始写了
                    while 1:
                        line = f.readline()                         # 继续遍历下一行
                        if line[0] == '\n':                         # 如果是空白行可以直接退出
                            break
                        hmm_seqac.append(line.split()[8])           # 没退出就继续执行，把序列号存在列表
                        count += 1                                  # 存一次加一次
                    break                                  # 遇到了空白行，连续执行退出了
            f.close()

        # 4.2把序列号写入txt
        with open('{path}/{ac}/0/{ac}_{pf}_hmm_uni_seqac'.format(path=hmm_out_path, ac=ac, pf=pf), 'w') as o:
            for seqac in hmm_seqac:
                o.write(seqac + '\n')
            o.write('# ' + str(count))
            o.close()

        # 5.提取序列，同上一个文件的步骤（三），就是用blast+去提取序列们，合成一个fasta
        # 如果出现 Error:[blastdbcmd]Skipped # 是正常的，因为我在上面4.2写序列号时，把序列号总数写入了最后一行，用#开头，blastdbcmd就会这样报错
        os.system('blastdbcmd -db {db} -entry_batch {path}/{ac}/0/{ac}_{pf}_hmm_uni_seqac>\
                  {path}/{ac}/0/{ac}_{pf}_hmm_uni_seqs'
                  .format(db=db, path=hmm_out_path, ac=ac, c=count, pf=pf))



def iter_customize(db, hmm_db,
                   hmm_out_path,
                   initial_evalue, times, start_times, expend, current_evalue,
                   onefamily_seqacs):

    # 我只是想看看
    # for s in range(start_times, start_times+times):
    #     current_evalue = initial_evalue * (expend ** s)  # 乘法的角度。乘一次，循环一次。第几次循环，就对应乘几次。
    #     print(float(current_evalue))

    # 【然后】重复：sto-hmm-hmmsearch得out-序列号-fa  # 自定义拓宽阈值迭代的部分
    for ac in onefamily_seqacs:

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓接下来开始迭代这个过程↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        print('【开始】对{ac}_{pf}进行迭代'.format(ac=ac, pf=pf))

        for s in range(start_times, start_times+times):            # 第几次开始，一共循环几次。

            try:  # 创建一下，要是有就报错略过。
                os.makedirs(hmm_out_path + '/{ac}/{s}/'.format(ac=ac, s=s))  # 创建多级文件夹的命令。
            except:
                pass
            print('【第{s}次】'.format(s=s))

            # 1.前体文件是{path}/{ac}/？/，用它制作序列比对
            print('【开始】mafft/hmmalign多序列比对'.format(ac=ac))
            ex_s = s - 1       # s是当前次数，s-1表示前一次   # 这里不是断点而是正常循环的话，只需要正常接收current_evalue的循环数值
            # 由于mafft给出的fa格式的比对过于巨大，40G，导致hmmbuild崩溃，因此，我们使用hmmalign替代此过程。
            # os.system('mafft --retree 1 --maxiterate 0 --nofft --parttree \
            #           --quiet \
            #           {inpath}/{ac}/{ex_s}/{ac}_PF00083_hmm_uni_seqs > {outpath}/{ac}/{s}/{ac}_PF00083.fa'
            #           .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s, ex_s=ex_s))
            # 后缀名.fa我就不改了，实际上是sto格式。
            os.system('hmmalign --outformat stockholm \
                      -o {outpath}/{ac}/{s}/{ac}_{pf}.fa \
                      {inpath}/{ac}/{ex_s}/{ac}_{pf}.hmm {inpath}/{ac}/{ex_s}/{ac}_{pf}_hmm_uni_seqs'
                      .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s, ex_s=ex_s, pf=pf))
            print('mafft/hmmalign结束')  # 序列比对

            # 2.以序列比对为前体
            print('【开始】hmmbuild以.fa构建隐马模型')
            # 注意写你自己的hmmbuild地址
            # 在结尾加上 > /dev/null，以“删除”所有的屏幕输出
            os.system(
                'hmmbuild --cpu 30 \
                {outpath}/{ac}/{s}/{ac}_{pf}.hmm \
                {inpath}/{ac}/{s}/{ac}_{pf}.fa'
                .format(ac=ac, inpath=hmm_out_path, outpath=hmm_out_path, s=s, pf=pf))  # 用该序列比对结果生成hmm
            print('hmmbuild建模结束')

            # 3.以该hmm作为前体
            print('【开始】hmmsearch搜索')
            # 初次实验发现，current_evalue是定值的情况下，也能逐步扩大数据集，因为hmmpx本来就越来越小。
            # current_evalue = initial_evalue + expend * s    # 加法的角度。加一次，循环一次。
            current_evalue = initial_evalue * (expend ** s)     # 乘法的角度。乘一次，循环一次。第几次循环，就对应乘几次。
            # 注意写你自己的hmmsearch地址
            os.system('hmmsearch --noali --notextw --cpu 30 \
                      -E {ce} \
                      -o {outpath}/{ac}/{s}/{ac}_{pf}_hmm_uni.out \
                      {inpath}/{ac}/{s}/{ac}_{pf}.hmm \
                      {db}'
                      .format(inpath=hmm_out_path, outpath=hmm_out_path, ac=ac, db=hmm_db,
                              ce=current_evalue, s=s, pf=pf))  # 用hmmsearch搜出了out
            print('hmmsearch搜索结束')

            # 4.1把out中的序列号的部分提取出来。
            count = 0  # 计序列号数
            hmm_seqac = []  # 存放序列号的列表
            with open('{path}/{ac}/{s}/{ac}_{pf}_hmm_uni.out'.format(path=hmm_out_path, ac=ac, s=s, pf=pf)) as f:  # 打开out
                while 1:  # 遍历每一行
                    line = f.readline()  # 按行读取避免内存溢出
                    if line[0:11] == '    -------':  # 当你遇到这样的行时，证明你可以开始写了
                        while 1:
                            line = f.readline()  # 继续遍历下一行
                            if line[0] == '\n' or line == '  ------ inclusion threshold ------\n':  # 如果是空白行或穷尽了，直接退出
                                break
                            hmm_seqac.append(line.split()[8])  # 没退出就继续执行，把序列号存在列表
                            count += 1  # 存一次加一次
                        break  # 遇到了空白行，连续执行退出了
                f.close()

            # 4.2把序列号写入txt
            with open('{path}/{ac}/{s}/{ac}_{pf}_hmm_uni_seqac'.format(path=hmm_out_path, ac=ac, s=s, pf=pf), 'w') as o:
                for seqac in hmm_seqac:
                    o.write(seqac + '\n')
                o.write('# ' + str(count))
                o.close()

            # 5.提取序列，同上一个文件的步骤（三）用blast+去提取序列们，合成一个fasta
            os.system('blastdbcmd -db {db} \
                      -entry_batch {inpath}/{ac}/{s}/{ac}_{pf}_hmm_uni_seqac>\
                      {outpath}/{ac}/{s}/{ac}_{pf}_hmm_uni_seqs'
                      .format(db=db, inpath=hmm_out_path, outpath=hmm_out_path, ac=ac, s=s, pf=pf))



if __name__ == '__main__':
    # local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    local_path = '/data1/kzw'   # 187

    # pf = 'PF00083'
    pf = sys.argv[1]    # 在命令行中接收参数

    db = '{lp}/repfam/{pf}/{pf}_uniprot_seq/db/{pf}_uniprot_seqs'.format(lp=local_path, pf=pf)  # 我的uni的blast的数据库，但hmm需要用fasta的格式的
    hmm_db = '{lp}/repfam/{pf}/{pf}_uniprot_seq/{pf}_uniprot_seqs.fasta'.format(lp=local_path, pf=pf)  # hmm可用的fasta数据合集
    seqana_bpseqs_path = '{lp}/repfam/{pf}/{pf}_stseq_ana'.format(lp=local_path, pf=pf)  # 第一次用blastp产生的衍生分析文件地址 'O97467_PF00083_bpseqs.fasta'
    hmm_out_path = '{lp}/repfam/{pf}'.format(lp=local_path, pf=pf)  # 结果文件保存的地址

    initial_evalue = 10**-150  # 初始阈值，根据83家族O97467序列私设。
    times = 30  # 进入迭代后循环的次数 ************** 注意是次数
    start_times = 1  # 如果中途中断过，想继续跑，请在此处【自选】从第几次开始。默认1,第1号迭代。******* 0号迭代由nitial_attempts函数控制，不在这里设置

    # 第一次实践结果：把current_evalue搞成定值，也能逐步增加序列数量。不需要expend。
    # 用的-20次幂，因为hmm本来就越来越大，不需要扩大阈值去扩增数量，反而需要减小阈值，控制序列数量别太多。
    # current_evalue = 10**-20 # 如果用定值
    # 第二次实践结果：current_evalue是定值的时候，没几次循环数据集便不再增长了。所以，还是得用expend，而且是乘以+次幂

    # expend = 10**-100  # 迭代时阈值放宽的幅度。加法的角度。
    current_evalue = 0  # 当前循环中使用的e值。会在循环中自动计算。 # 加法的角度才使用，目前这个版本不需要这个变量，但懒得删，所以随便设一个


    expend = 10**5  # 迭代时阈值放宽的幅度。乘法的角度。 # 对不起单词拼错了，将错就错吧

    # onefamily_seqacs = ['O97467']  # 本次只会用到这一个,在这个目前的83家族中。后续可以通过for进行添加。
    seqac = sys.argv[2]
    onefamily_seqacs = [seqac]

    # 如果是从断点开始，记得要把初次的迭代注释掉哇。
    nitial_attempts(db, hmm_db, seqana_bpseqs_path,
                    hmm_out_path,
                    initial_evalue,
                    onefamily_seqacs)       # 0号梯度，初次迭代
    iter_customize(db, hmm_db,
                   hmm_out_path,
                   initial_evalue, times, start_times, expend, current_evalue,
                   onefamily_seqacs)       # 1，2，3……号迭代


# 此文件曾用方案：

#（一）jackhmmer
# 在实践过程中，阈值总是难以把握的，默认E10的阈值会导致三次迭代完成整个80w数据。E1e-80会导致数据卡在15w。T更是很迷的阈值，不建议使用。
'''
jackhmmer 
-N 50 迭代五十次
-o final.out 最终总输出
-A final.sto 最终比对
--chkhmm jack 每次hmm前缀
--chkali jack 每次sto前缀
--enone 不要有效序列的确定，也即是，所有的都是有效序列
--notextw 最终out不限制每行字数
--noali 最终out不要比对部分
-E 0.1 阈值
--cpu 10 使用十个cpu核
O97467.fasta /disk2/kzw/repfam/PF00083/PF00083_uniprot_seq/PF00083_uniprot_seqs.fasta
'''
# 如使用jackhmmer，则从z4_1/z4_2之后都要改写，直接转到phmm程序即可
# 在自写的phmm中，我们会对每一阶梯的sto中的序列随机抽取与第一次迭代相同数量的序列，进行建模。
# 依然可以查看在自己的建模中序列的hmmpx，虽然这一点在jack迭代过程中已经被展示。
# 接下来，就是导出其viterbi的比对/次优比对，进而，我们对其进行sp，me打分，以及rmsd。

#（二）目前所用的，最新改进
# 把mafft过程全变成hmmalign 导出为sto格式的比对