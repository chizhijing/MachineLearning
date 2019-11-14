# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:24:47 2019

@author: chizj
"""
from common import get_train_data
my_stopwords="a : “ ” , v the its and as on 's ( ) : % . based".split(' ')



if __name__=='__main__':
    author_list=['li_guo', 'bo_shen', 'di_wang', 'long_wang', 'qiang_xu', 
                 'xiang_wang', 'changming_liu', 'kenji_kaneko', 'guohua_chen', 'hai_jin', 
                 'jia_li', 'guoliang_li', 'lan_wang', 'alessandro_giuliani', 'jiang_he', 
                 'xiang_gao', 'jianping_wu', 'peng_shi', 'feng_wu', 'jing_zhu']
    train_data,train_tag=get_train_data(author_list)

#    au='li_guo'
    
    from base import PaperGraph
    
    pg_dict={}
    for au in author_list:
        pg=PaperGraph(au,train_data[au],train_tag[au])
        g1=pg.get_graph_by_coauthor(pg.origin_graph)
        pg.get_score(g1)
        pg_dict[au]=(pg,g1)
        
    for au,v in pg_dict.items():
        v[0].get_score(v[1])
        
