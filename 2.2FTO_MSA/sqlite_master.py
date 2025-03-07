import sqlite3
import numpy as np
import pandas
from functools import lru_cache
from io import StringIO
from Graph import *
# ==========================================
#  该程序用于对数据库进行连接和查询
# ==========================================

class sql_master():
    def __init__(self,graph,db="",mode = 'memory'):
        if db =="":
            # 创建一个内存数据库用于建图或合图过程中即时查询节点前后关系
            self.conn = sqlite3.connect(":memory:")
            innerlinkset=[]
            for i in graph.linkset.keys():
                innerlinkset.append((i[0],i[1],graph.linkset[i]))
            df = pandas.DataFrame(innerlinkset,columns=['startnode','endnode','weight'])
            cursor = self.conn.cursor()
            # 建立新表
            sql = '''CREATE TABLE link(startnode int({0}),endnode int({0}),weight int({0}));'''.format(int(graph.fra)+3)
            cursor.execute(sql)

            sql = '''CREATE INDEX sin ON link (startnode);'''
            cursor.execute(sql)
            
            sql = '''CREATE INDEX ein ON link (endnode);'''
            cursor.execute(sql)

            df.to_sql('link', self.conn, if_exists='append', index=False)

            self.conn.commit()
        elif mode=='memory':
            # 连接一个数据库
            
            

            tmpconn = sqlite3.connect(db)
            self.conn = sqlite3.connect(":memory:")
            
            # 将现有数据库文件的内容导入到内存数据库
            with tmpconn, self.conn:
                for line in tmpconn.iterdump():
                    self.conn.execute(line)
        elif isinstance(mode,list):
            # 连接到磁盘数据库
            disk_conn = sqlite3.connect(db)
            disk_cursor = disk_conn.cursor()

            # 创建内存数据库
            self.conn = sqlite3.connect(':memory:')
            memory_cursor = self.conn .cursor()

            # 要复制的表列表
            tables_to_copy = mode

            for table in tables_to_copy:
                # 获取表的创建语句
                disk_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
                create_table_sql = disk_cursor.fetchone()[0]
                
                # 在内存数据库中创建表
                memory_cursor.execute(create_table_sql)
                
                # 从磁盘数据库中读取数据
                disk_cursor.execute(f"SELECT * FROM {table}")
                rows = disk_cursor.fetchall()
                
                # 获取列名
                column_names = [description[0] for description in disk_cursor.description]
                columns = ', '.join(column_names)
                placeholders = ', '.join(['?'] * len(column_names))
                
                # 将数据插入到内存数据库中
                memory_cursor.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", rows)

            # 提交更改
            self.conn.commit()

            # 关闭连接
            disk_conn.close()
            pass
        else:
            self.conn = sqlite3.connect(db)

    def __enter__(self):
        return self

    def __exit__(self):
        self.conn.close

    def find_virus_sub(self,gid,node_id,first_number_bite_of_ONM=32,all_bite_of_ONM=64,first_number_bite_of_OSM=16,all_bite_of_OSM=32):
        # 查询短序列节点包含的病毒
        
        cursor = self.conn.cursor()
        query = 'SELECT virus FROM vset'+str(gid)+' WHERE node_id IN ({})'.format(','.join(['?']*len(node_id)))
        nids =GRAPH.get_second_number(node_id,first_number_bite_of_ONM,all_bite_of_ONM)
        cursor.execute(query, [int(item) for item in nids])
        
        results=cursor.fetchall()
        virus_list=[]
        allsize=0
        for result in results:
            v = np.frombuffer(result[0], dtype=np.uint32)
            allsize+=v.size
            virus_list.append(v)
        virus = np.empty(allsize, dtype=np.uint32)
        size_now=0
        for v in virus_list:
            size = v.size
            virus[size_now:size_now+size]=v
            size_now+=size
            

        alist = GRAPH.save_numbers(gid,GRAPH.get_first_number(virus,first_number_bite_of_OSM,all_bite_of_OSM),16,32)
        blist = GRAPH.get_second_number(virus,first_number_bite_of_OSM,all_bite_of_OSM)
        VS=list(zip(alist, blist))
        return VS
    
    def findSequenceSource(self,nodes,first_number_bite_of_ONM=32,all_bite_of_ONM=64,first_number_bite_of_OSM=16,all_bite_of_OSM=32):
        results = list()
        nodes = np.array(nodes)
        op = self.find_virus_sub
        # gid_of_nodes = GRAPH.get_ori_graph_id(nodes)
        gid_of_nodes = GRAPH.get_first_number(nodes,first_number_bite_of_ONM,all_bite_of_ONM)
        gidset = set(gid_of_nodes)
        for gid in gidset:
            query_nodes = nodes[np.where(gid_of_nodes==gid)[0]]
            results.extend(op(gid,query_nodes,first_number_bite_of_ONM,all_bite_of_ONM,first_number_bite_of_OSM,all_bite_of_OSM))


        matrix = np.array(results,order='F',dtype=np.uint32).T

        return matrix

    


    def metadata(self,vname):
        cursor = self.conn.cursor()
        sql = "select Is_complete,Is_high_coverage,N_Content from metadata where Virus_name = '{}'".format(vname)
        
        cursor.execute(sql)
        relist = cursor.fetchone()
        return relist
    



