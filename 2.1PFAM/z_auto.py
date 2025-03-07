# Z4_2的自动化调用


import os



# pf_li = ['PF00083', 'PF00892', 'PF00535']
pf_li = ['PF00083', 'PF00892', 'PF00535']

lili = ['seq_083_li', 'seq_892_li', 'seq_535_li']

# seq_083_li = ['O97467', 'G4TS85', 'A0A0H2VG78' ]
seq_083_li = ['A0A0H2VG78']
seq_892_li = ['D7A5Q8', 'A0A0E7XN74', 'A0A0H2UQH5']
seq_535_li = ['Q04TP4', 'A0A0H3JNB0', 'B7SY86']
for i in range(0, 3):
    for j in range(len(eval(lili[i]))):
        print('nohup python3 z4_2_hmmbuild_cluster_hmm.py {pf} {seqac} > z4_2_{pf}_{seqac}.log 2>&1 &'
              .format(pf=pf_li[i], seqac=eval(lili[i])[j]))

# nohup python3 z_auto.py > z_auto.log 2>&1 &