# 本程序计算每一个阶梯px对应的seqs们的所有的pdb，并计算他们的rmsd。针对random文件夹。
# pdb直接用AF数据库网站里现成的就行。AF把市面上大部分80%的蛋白质都预测了结构存放在数据库，可以直接利用url下载。
# 每一个迭代文件0/1/2/3等等。0中存放的比对fa，对应着0中的hmmpx，对应着0的random中的sp、me、rmsd。但是，却对应着-1步的seqsfasta集合。
import time
import requests
import os
import sys



def download_seqs(pf, ac, hmm_out_path, i):   # 家族号，结构序列号，ana地址，输出地址，迭代次数

    # 把每个迭代文件包里的random里的seqac都下载一下pdb，然后放在pdb文件夹里。

    # os.system('rm -r {hmm_out_path}/{ac}/{s}/random/pdb/'.format(hmm_out_path=hmm_out_path, ac=ac, s=i))  # 删除测试

    try:  # 创建一下，要是有就报错略过。
        os.makedirs(hmm_out_path + '/{ac}/{s}/random/pdb/'.format(ac=ac, s=i))  # 创建多级文件夹的命令。
    except:
        pass
    # 直接取该迭代文件夹内的seqac
    random_seqac_path = hmm_out_path + '/{ac}/{s}/random/{ac}_{pf}_hmm_uni_seqac'.format(ac=ac, pf=pf, s=i)
    with open(random_seqac_path) as f:
        while 1:
            line = f.readline()
            seqac = line.strip()
            if not line:  # 如果是空白行可以直接退出
                break
            url = 'https://alphafold.ebi.ac.uk/files/AF-{seqac}-F1-model_v4.pdb'.format(seqac=seqac)   # 网址固定格式
            response = requests.get(url)  # 这里存放的是一个实例，而且是二进制的，下面必须用wb
            if response.status_code == 200:           # 如果这个网址存在，写入文件
                with open(hmm_out_path + '/{ac}/{s}/random/pdb/{seqac}.pdb'.format(ac=ac, s=i, seqac=seqac), 'wb') as o:
                    for data in response:
                        o.write(data)
                    o.close()



def rmsd(ac_pdb_path, ac, hmm_out_path, i):   # 基准结构=最初的有结构序列，最初有结构序列，outpath，迭代次数的文件夹。

    # 计算rmsd的
    pdb_path = hmm_out_path + '/{ac}/{s}/random/pdb'.format(ac=ac, s=i)  # 首先，每一个迭代文件夹的pdb的存放位置
    filenames_pdb_path_li = os.listdir(pdb_path)    # 取这个位置的所有的文件名
    i_rmsd = 0        # 该迭代文件下的所有的rmsd的总分

    for filename in filenames_pdb_path_li:        # 对于每一份文件名

        seqac = filename.split('.')[0].strip()            # 取.前面的seqac

        with open('z5_6_pymol_align.py', 'w') as o:
            o.write('import pymol' + '\n')
            o.write('cmd.load(\'{ac_pdb_path}\')'.format(ac_pdb_path=ac_pdb_path) + '\n')
            o.write('cmd.load(\'{pdb_path}/{seqac}.pdb\')'.format(pdb_path=pdb_path, seqac=seqac) + '\n')
            o.write('print(pymol.fitting.align(\'{seqac}\', \'{ac}_0\')[0])'.format(seqac=seqac, ac=ac) + '\n')
            o.write('cmd.quit()' + '\n')
            o.close()
        os.system('pymol z5_6_pymol_align.py > z5_6.log')   # -c就是不弹出图形界面。但最新版好像命令有变化？报错了
        with open('z5_6.log') as f:
            lines = f.readlines()
            # print(lines)
            rmsd = lines[-1].strip()
        os.system('rm z5_6_pymol_align.py')
        os.system('rm z5_6.log')
        print(filename, rmsd)
        i_rmsd = i_rmsd + float(rmsd)

        # 测试
        # cmd.load(ac_pdb_path)  # pymol加载基准pdb
        # cmd.load(pdb_path + '/{seqac}.pdb'.format(seqac=seqac))  # pymol加载比较pdb
        # i_rmsd += pymol.fitting.align(seqac, ac)[0]  # 比

    rmsd = i_rmsd / len(filenames_pdb_path_li)      # 总分要除以文件数目，得出一个rmsd的平均分。rmsd只能两两比对，以ac为基准。

    return rmsd



if __name__ == '__main__':
    local_path = '/home/handsomekk'
    # local_path = '/disk2/kzw'
    # local_path = '/data1/kzw'

    # pf = 'PF00083'
    pf = sys.argv[1]

    hmm_out_path = '{lp}/repfam/{pf}'.format(lp=local_path, pf=pf)  # 结果文件保存的地址

    # onefamily_seqacs = ['G4TS85']  # 本次我只会用到这一个,在这个目前的83家族中，针对这一个做一下。后续可以通过for进行添加。
    seqac = sys.argv[2]
    onefamily_seqacs = [seqac]

    times = int(sys.argv[3])

    for ac in onefamily_seqacs:  # 对于该家族中的每一条结构序列

        # 先把基准ac的pdb下载一下。并且给基准改个名
        ac_pdb_path = hmm_out_path + '/{ac}/{ac}_0.pdb'.format(ac=ac)  # 为了避免重名错误，因此，改l个名。
        # # 先把基准ac的pdb下载一下。
        # ac_url = 'https://alphafold.ebi.ac.uk/files/AF-{ac}-F1-model_v4.pdb'.format(ac=ac)  # 网址固定格式
        # response = requests.get(ac_url)  # 这里存放的是一个实例，而且是二进制的，下面必须用wb
        # if response.status_code == 200:  # 如果这个网址存在
        #     with open(ac_pdb_path, 'wb') as o:  # 打开写入
        #         for data in response:
        #             o.write(data)
        #         o.close()

        # # 下载所有的random里的pdb
        # for i in range(1, times+1):  # z4.2.0里做了几个times/节点/阶梯就range几+1。从1开始。
        #     print(time.ctime(), 'rmsd下载开始', i)
        #     download_seqs(pf, ac, hmm_out_path, i)    # 下载每个迭代文件包里的random里的seqac们的pdb
        #     print(time.ctime(), 'rmsd下载结束', i)

        # 计算rmsd。用pymol xxx.py
        rmsd_li = ['rmsd']
        for i in range(1, times+1):
            print(time.ctime(), 'rmsd计算开始', i)
            rmsd_li.append(rmsd(ac_pdb_path, ac, hmm_out_path, i))    # 计算每一个迭代文件包里的random里的平均rmsd，存到li
            print(time.ctime(), 'rmsd计算结束', i)
        with open(hmm_out_path + '/{ac}/get_rmsd'.format(ac=ac), 'w') as o:
            for every_rmsd in rmsd_li:
                o.write(str(every_rmsd) + '\n')
            o.close()