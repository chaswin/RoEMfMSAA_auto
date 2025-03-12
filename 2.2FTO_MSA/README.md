## 本程序是随机扰动实验程序
主程序DAG_PHMM.py。auto_DAG.py是方便我自己用来自动调用主程序整的脚本。 \
### 扰动策略：
仅针对ms函数进行扰动，插入random的位置已经由 “ # # ” 标注出来 \
\
如下示例：\
\
random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1)) \
因此，每个地方的random值都不一样 \
其中，所有的都统一设置为：m_global_random_low=0.2，m_global_random_up=0.4 \
该策略可保证更优解出现的概率最大，并且在数据规模增大时，比对精确度的提高更明显。
### FTO_MSA代码暂不开源，敬请期待
本包原版仅用了单线程建图，10万条序列跑一次要5h左右。 \
新版内部改为多线程，效率应有提升，但本研究并未尝试。
### 目前本包仅提供实验结果数据表格
