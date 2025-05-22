## 本程序是随机扰动实验程序
主程序DAG_PHMM.py。auto_DAG.py是方便我自己用来自动调用主程序而整的脚本。 \
### 扰动策略：
仅针对ms函数进行扰动，插入random的位置已经由 “ # # ” 标注出来 \
\
如下示例：\
\
random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1)) \
所以，每个地方的random值都是不一样的 \
建议统一设置为：m_global_random_low=0.2，m_global_random_up=0.4 \
该策略可保证更优解出现的概率最大。
### FTO_MSA代码暂不开源，敬请期待
本包原版仅用了单线程建图，10万条序列跑一次要5h左右。 \
本包新版内部改为多线程，效率有很大提升，10万条序列跑一次只需1h多。
### 相关文献和作者请联系如下
FTO-MSA算法的相关信息请参考以下文献： \
[1]赖潇.片段拓扑序引导的大规模相似基因多序列比对算法[D].吉林大学,2022.DOI:10.27162/d.cnki.gjlin.2022.001149. \
[2]Xiao Lai ,Haixin Luan ,Pu Tian .Accurate Multiple Sequence Alignment of Ultramassive Genome Sets.[J].bioRxiv,2024:p.2024.09.22.613454
### 目前本文件夹下仅提供实验结果数据表格
