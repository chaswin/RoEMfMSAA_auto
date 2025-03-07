# 本程序是：自动生成随机扰动的范围的上下限，并启动DAG，按照给定次数无限运行。
# 并且包含了最后的结果文件精简步骤。

import os
import sys
import numpy as np



# 无限循环跑，要跑第几次到第几次的？
tslow = int(sys.argv[1])     # 下限
tsup = int(sys.argv[2])      # 上限

# 参数的上下限范围
i_global_random_low = float(sys.argv[3])                 # 初始参数，全局变量，随机数下限
i_global_random_up = float(sys.argv[4])                 # 初始参数，全局变量，随机数上限
m_global_random_low = float(sys.argv[5])                 # 更新参数，全局变量，随机数下限
m_global_random_up = float(sys.argv[6])                 # 更新参数，全局变量，随机数上限

# 本地地址
# local_path = '/home/handsomekk'
# local_path = '/data1/kzw'         # 187
local_path = '/data/kzw'            # 186

# 数据集ID，原
# subgraphID = '5'            # 5   这里必须纯数字
# subgraphID = '90'           # 50000
subgraphID = sys.argv[7]    # ID

# fasta数据集地址
# fastapath = '/home/handsomekk/repfam/fto/5.fasta'         # 30本地5条
# fastapath = '/data1/kzw/try/5.fasta'                      # 187节5条
# fastapath = '/data1/kzw/try/{id}.fasta'.format(id=subgraphID)       # 187
fastapath = '/data/kzw/try/{id}.fasta'.format(id=subgraphID)        # 186


# 扰动类型
# folder = subgraphID + '/all_random_same'       # 如5/all_random，是数据集5对应的，全扰动类型
# folder = subgraphID + '/ini_random'       # 初始扰动
folder = subgraphID + '/mstep_random'     # 更新扰动


# 对照组
ifnorandom = str(sys.argv[8])
if ifnorandom == 'yes':
    norandom_outpath = '{lp}/{id}'.format(lp=local_path, id=subgraphID)
    os.system('python3 DAG_PHMM_norandom.py {fp} {op}/deep/ {id} 0 0 0 0 > {id}_norandom.log'
              .format(fp=fastapath, op=norandom_outpath, id=subgraphID))
    os.system('mv {op}/deep/zip20/ali_result_tr1/zipali.npy {op}'.format(op=norandom_outpath))
    os.system('rm -r {op}/deep'.format(op=norandom_outpath))
    os.system('mv ./{id}_norandom.log {op}'
              .format(id=subgraphID, op=norandom_outpath))
    print('对照组 is over.')


# 开始无限循环跑，要跑第几次到第几次的？
for ts in range(tslow, tsup+1):

    # 每次的输出地址
    # outpath = '{lp}/repfam/fto/{folder}/5-{ts}/deep/'.format(lp=local_path, folder=folder, ts=ts)            # 30
    outpath = '{lp}/{folder}/{id}-{ts}/deep/'.format(lp=local_path, folder=folder, id=subgraphID, ts=ts)       # 其他


    # 自动的-扰动范围上下限生成器 !!! [注意，不能是负数，会nan无限循环]
    # # （1）当你希望 初始-更新 参数一致的情况下
    # ch_bo_01 = np.random.choice([0, 1])   # 随机选择其一。0表示<1，1表示>1。这样使<1的数占比更大。choice both 0or1
    # if ch_bo_01 == 0:
    #     ch_bo = round(np.random.uniform(0, 1), 5)                # 随机浮点数，np左闭右开
    # else:
    #     ch_bo = round(np.random.uniform(1, 2), 5)                # 随机浮点数，np左闭右开
    # i_global_random_low = ch_bo                        # 初始参数，全局变量，随机数下限
    # i_global_random_up = round(ch_bo + 0.00002, 5)     # 初始参数，全局变量，随机数上限
    # m_global_random_low = ch_bo                        # 更新参数，全局变量，随机数下限
    # m_global_random_up = round(ch_bo + 0.00002, 5)     # 更新参数，全局变量，随机数上限
    # # （2）当你希望 初始-更新 参数各不相同的情况下
    # ch_i_01 = np.random.choice([0, 1])  # 给初参，随机选择其中一个。0表示<1，1表示>1。
    # ch_m_01 = np.random.choice([0, 1])  # 给更参，随机选择其中一个。0表示<1，1表示>1。
    # if ch_i_01 == 0:
    #     ch_i = round(np.random.uniform(0, 1), 5)  # 随机浮点数，np左闭右开
    # else:
    #     ch_i = round(np.random.uniform(1, 6), 5)  # 随机浮点数，np左闭右开
    # if ch_m_01 == 0:
    #     ch_m = round(np.random.uniform(0, 1), 5)  # 随机浮点数，np左闭右开
    # else:
    #     ch_m = round(np.random.uniform(1, 6), 5)  # 随机浮点数，np左闭右开
    # i_global_random_low = ch_i                          # 初始参数，全局变量，随机数下限
    # i_global_random_up = round(ch_i + 0.00002, 5)       # 初始参数，全局变量，随机数上限
    # m_global_random_low = ch_m                          # 更新参数，全局变量，随机数下限
    # m_global_random_up = round(ch_m + 0.00002, 5)       # 更新参数，全局变量，随机数上限
    # # （3）当你希望，每个添加扰动数字的地方都不一样！
    # i_global_random_low = 0                 # 初始参数，全局变量，随机数下限
    # i_global_random_up = 0                 # 初始参数，全局变量，随机数上限
    # m_global_random_low = 0                 # 更新参数，全局变量，随机数下限
    # m_global_random_up = 0.2                 # 更新参数，全局变量，随机数上限

    # print(i_global_random_low,i_global_random_up,m_global_random_low,m_global_random_up)

    # 执行
    os.system('python3 DAG_PHMM.py {fp} {op} {id} {ilow} {iup} {mlow} {mup} > {ilow}-{iup}_{mlow}-{mup}.log{id}'
              .format(fp=fastapath, op=outpath, id=subgraphID, ilow=i_global_random_low, iup=i_global_random_up,
                      mlow=m_global_random_low, mup=m_global_random_up))


    # 结果文件简化，只留zipali和log。DAG可以放在任一固定独立文件夹。
    file = outpath.split('/')[-3]                       # 如5-1，是数据集对应的梯度批次

    # 每次的梯度批次层次的输出地址
    # path = '{lp}/repfam/fto/{folder}/{file}'.format(lp=local_path, folder=folder, file=file)      # 30
    path = '{lp}/{folder}/{file}'.format(lp=local_path, folder=folder, file=file)                 # 187、186

    os.system('mv {path}/deep/zip20/ali_result_tr1/zipali.npy {path}'.format(path=path))
    os.system('rm -r {path}/deep'.format(path=path))
    os.system('mv ./{ilow}-{iup}_{mlow}-{mup}.log{id} {path}'
              .format(path=path, ilow=i_global_random_low, iup=i_global_random_up,
                      mlow=m_global_random_low, mup=m_global_random_up, id=subgraphID))

    print('finished {ts} times.'.format(ts=ts))


# 本程序使用方法：在终端
#       python  xxx.py 次数下限标号 次数上限标号 ilow iup mlow mup ID 是否对照 > ID日志参数范围mlow_mup内自动跑20次
# nohup python3 auto_DAG.py 1 20 ilow iup mlow mup ID no > IDauto0_0.2.log 2>&1 &

# mlow = 0
# for i in range(1, 101, 20):
#     mup = round(mlow + 0.2, 2)
#     print('nohup python3 auto_DAG.py {i} {i2} 0 0 {mlow} {mup} {id} no > {id}_{i}_{i2}_auto_{mlow}_{mup}.log 2>&1 &'
#           .format(i=i, i2=i+19, id='100', mlow=mlow, mup=mup))
#     mlow = round(mlow + 0.2, 2)