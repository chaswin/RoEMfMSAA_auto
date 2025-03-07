import glob
import re
import numpy as np

def sp_entro(inpath):
    Binpath = inpath
    # zip_ali = np.load(Binpath,allow_pickle=True).tolist()
    zip_ali = np.load(Binpath,allow_pickle=True)
    all_entropy = 0
    negative_score = 0
    positive_score=0
    sumnum = zip_ali[-1]
    length = len(zip_ali)-1
    for i in range(length):
        column = zip_ali[i]
        n_score,p_score,entropy = calculate_column(column,sumnum)
        negative_score += n_score
        positive_score += p_score
        all_entropy += entropy

    return positive_score,negative_score,all_entropy


def calculate_column(column,length):
    # print(column)
    num_dict={}
    sp_num=0
    for i in column[1:]:
        num_dict[i[0]]=len(i[1])
        sp_num+=num_dict[i[0]]
    num_dict[column[0]]=length-sp_num
    if set(num_dict.keys())==set([0]):
        return 0,0,0
    
    # 更新 num_dict 中的简并碱基
    for n in [n for n in num_dict.keys() if n > 4]:
        deg_base = degenerate_base_dict[n]
        for s in deg_base:
            num_dict[s] = num_dict.get(s, 0) + num_dict[n] / len(deg_base)
        del num_dict[n]

    # 计算得分
    n_score = 0
    p_score = 0
    Klist = num_dict.keys()
    doneset=set()
    for n in Klist:
        num = num_dict[n]
        n_score += (num - 1) * num / 2 * negative[n][n]
        p_score += (num - 1) * num / 2 * positive[n][n]
        doneset.add(n)
        for b in Klist-doneset:
            bnum = num_dict[b]
            n_score += num * bnum * negative[n][b]
            p_score += num * bnum * positive[n][b]


    # 计算熵
    entropy = 0
    for n in Klist:
        num = num_dict[n]
        if num != 0:
            P = num / length
            entropy += -P * np.log(P)

    return n_score,p_score,entropy

def calculate_std(data):
    mean = np.mean(data)
    positive_errors = data[data > mean] - mean
    negative_errors = mean - data[data < mean]
    positive_variance = np.mean(positive_errors ** 2)
    negative_variance = np.mean(negative_errors ** 2)
    positive_std_dev = np.sqrt(positive_variance)
    negative_std_dev = np.sqrt(negative_variance)
    std = np.std(data)
    return positive_std_dev,negative_std_dev,std

degenerate_base_dict = {0:[0],1:[1],2:[2],3:[3],4:[4],5: [1, 4], 6: [3, 2], 7: [1, 3], 8: [4, 2], 9: [4, 3], 10: [1, 2], 11: [1, 2, 3], 12: [4, 2, 3], 13: [4, 1, 3], 14: [4, 1, 2], 15: [1, 2, 3, 4]}
# negative = {
#     1: {1:  0, 3: -2, 4: -1, 2: -2 ,0: -4},
#     3: {1: -2, 3:  0, 4: -2, 2: -1 ,0: -4},
#     4: {1: -1, 3: -2, 4:  0, 2: -2 ,0: -4},
#     2: {1: -2, 3: -1, 4: -2, 2:  0 ,0: -4},
#     0: {1: -4, 3: -4, 4: -4, 2: -4 ,0: -1},
#     } 
# positive= {
#     1: {1:  1, 3:  0, 4:  0, 2:  0 ,0:  0},
#     3: {1:  0, 3:  1, 4:  0, 2:  0 ,0:  0},
#     4: {1:  0, 3:  0, 4:  1, 2:  0 ,0:  0},
#     2: {1:  0, 3:  0, 4:  0, 2:  1 ,0:  0},
#     0: {1:  0, 3:  0, 4:  0, 2:  0 ,0:  0},
#     }
negative = {
    1: {1:  0, 3: -4, 4: -4, 2: -4 ,0: -2},
    3: {1: -4, 3:  0, 4: -4, 2: -4 ,0: -2},
    4: {1: -4, 3: -4, 4:  0, 2: -4 ,0: -2},
    2: {1: -4, 3: -4, 4: -4, 2:  0 ,0: -2},
    0: {1: -2, 3: -2, 4: -2, 2: -2 ,0: -1},
    } #nuc44
positive= {
    1: {1:  5, 3:  0, 4:  0, 2:  0 ,0:  0},
    3: {1:  0, 3:  5, 4:  0, 2:  0 ,0:  0},
    4: {1:  0, 3:  0, 4:  5, 2:  0 ,0:  0},
    2: {1:  0, 3:  0, 4:  0, 2:  5 ,0:  0},
    0: {1:  0, 3:  0, 4:  0, 2:  0 ,0:  0},
    } #nuc44

if __name__ == '__main__':
    # 选择本地地址
    # local_path = '/home/handsomekk'
    local_path = '/data1/kzw'
    # local_path = '/data/kzw'  # 186

    # 选择数据集名称 id
    # data = '5'              # 5
    data = ''           # 纯数字

    # 选择本次要保存的文件名
    save_file = '_ms-no_1-100'  # 本次计算的扰动类型

    # 选择本次要保存到的地址
    # save_path = '{lp}/repfam/fto/{data}/{sf}'.format(lp=local_path, data=data, sf=save_file)    # 30
    save_path = '{lp}/{data}/{sf}'.format(lp=local_path, data=data, sf=save_file)               # 其他


    with open(save_path, 'w') as o:


        # 计算no_random文件

        o.write(data + ',')     # 写入数据名称作表头

        # 选择文件地址
        # path = '{lp}/repfam/fto/{data}'.format(lp=local_path, data=data)        # 文件夹完整地址。30
        path = '{lp}/{data}'.format(lp=local_path, data=data)                   # 其他

        log = glob.glob('{path}/*norandom.log'.format(path=path))               # 文件夹之下的log，返回了一个列表
        with open(log[0]) as f:                                # 取列表第一个，因为只有一个
            fstr = f.read()                             # 读取文件内容为字符串
            probli = re.findall('Prob.*', fstr)         # 正则匹配prob，后面.任意字符匹配*数次。返回列表。
            px = probli[-1].split()[-1]                 # 取得最后一次的概率分数，并切片，
            o.write(px + ',')   # 写入px
            f.close()
        n_score, p_score, entropy = sp_entro('{path}/zipali.npy'.format(path=path))
        o.write(str(n_score) + ',')     # 写入sp，罚分，熵
        o.write(str(p_score) + ',')
        o.write(str(entropy) + ',')
        o.write('0' + ',')
        o.write('0' + ',')
        o.write('0' + ',')
        o.write('0' + '\n')


        # 计算其他random文件夹

        # 选择数据集与扰动类型
        # folder = '{data}/all_random_same'.format(data=data)        # 选一个文件夹，你要计算的那一批次的
        # folder = '{data}/ini_random'.format(data=data)        # 选一个文件夹
        folder = '{data}/mstep_random'.format(data=data)      # 选一个文件夹

        # 选择计算的文件范围
        for i in range(1, 101):

            # 选一下文件夹名称格式
            file = '{data}-{i}'.format(data=data, i=i)

            o.write(file + ',')  # 写入表头

            # 选择文件地址
            # path = '{lp}/repfam/fto/{folder}/{file}'.format(lp=local_path, file=file, folder=folder)      # 30
            path = '{lp}/{folder}/{file}'.format(lp=local_path, file=file, folder=folder)                 # 其他

            log = glob.glob('{path}/*.log*'.format(path=path))  # 文件夹之下的log，返回了一个列表
            print(log)
            with open(log[0]) as f:  # 取列表第一个，因为只有一个
                fstr = f.read()  # 读取文件内容为字符串
                probli = re.findall('Prob.*', fstr)  # 正则匹配prob，后面.任意字符匹配*数次。返回列表。
                px = probli[-1].split()[-1]  # 取得最后一次的概率分数，并切片，
                o.write(px + ',')  # 写入px
                f.close()
            n_score, p_score, entropy = sp_entro('{path}/zipali.npy'.format(path=path))
            o.write(str(n_score) + ',')
            o.write(str(p_score) + ',')
            o.write(str(entropy) + ',')

            low_up = log[0].split('/')[-1].split('.log')[0]
            ilowup = low_up.split('_')[0]
            mlowup = low_up.split('_')[1]
            ilow = ilowup.split('-')[0]
            iup = ilowup.split('-')[1]
            mlow = mlowup.split('-')[0]
            mup = mlowup.split('-')[1]
            o.write(ilow + ',')
            o.write(iup + ',')
            o.write(mlow + ',')
            o.write(mup + '\n')


        o.close()


    # n_score, p_score, entropy = sp_entro('{path}/zip20/ali_result_tr1/zipali.npy'.format(path=path))
    # n_score,p_score,entropy = sp_entro('/state1/result_DATA_back_up/Graph_test2/new_far_test/1_40000/zip20/ali_result_tr1/zipali.npy')

    # print(n_score, p_score, entropy)