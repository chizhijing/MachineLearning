# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:52:30 2019

@author: chizj
"""
from PaperGraph import get_sample_data,get_valid_data
from common import pairwise_f1,pairwise_f1_new
import json

if __name__=='__main__':
    data,tag=get_sample_data()

    from PaperGraph import PaperGraph
    
    authors=data.keys()
    for author in authors:
        p_graph=PaperGraph(name=author)
        p_graph.set_paper_info(data[author])    
        con=p_graph.get_res1()
        f1=pairwise_f1({author:tag[author]},{author:con})
        print(author,'%.2f'%f1,len(tag[author]),len(con))
    
    for author in authors:
        p_graph=PaperGraph(name=author)
        p_graph.set_paper_info(data[author]) 
        p_graph.cal_node_pair_info()
        con=p_graph.get_res2()
        f1=pairwise_f1({author:tag[author]},{author:con})
        print(author,'%.2f'%f1,len(tag[author]),len(con))
    
    for author in authors:
        p_graph=PaperGraph(name=author)
        p_graph.set_paper_info(data[author]) 
        p_graph.cal_node_pair_info()
        con=p_graph.get_res3()
        f1=pairwise_f1({author:tag[author]},{author:con})
        print(author,'%.2f'%f1,len(tag[author]),len(con))
    
    num=280
    for i in range(num):
        for j in range(num):
            if j<=i:continue
            d=p_graph.dist1(p_graph.get_pid_at(i),p_graph.get_pid_at(j))
           
# 某一个作者的情况
    author='bo_shen'
    p_graph=PaperGraph(name=author)
    p_graph.set_paper_info(data[author]) 
    con=p_graph.get_res1()
    f1=pairwise_f1({author:tag[author]},{author:con})
    print(author,'%.2f'%f1)
    
    for pid in tag[author][2]:
        author_org=p_graph.node[pid]['author_org']
        if author_org!='':
            print(author_org)
    
    s1=p_graph.node[tag[author][25][3]]['author_org']
    s2=p_graph.node[tag[author][25][2]]['author_org']
    set1=set(re.split(r'[;,\s]\s*',s1))
    set2=set(re.split)
    
    
    
##################### 验证集 ################################     
    v_data=get_valid_data()
    res_dict={}
    for author in v_data.keys():
        p_graph=PaperGraph(name=author)
        p_graph.set_paper_info(v_data[author])    
        p_graph.cal_node_pair_info()
        con=p_graph.get_res2()
        res_dict[author]=con
    json.dump(res_dict, open('result/pg_res2.json', 'w', encoding='utf-8'), indent=4)