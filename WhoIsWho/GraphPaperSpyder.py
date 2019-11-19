# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:52:30 2019

@author: chizj
"""


if __name__=='__main__':
    data,tag=get_sample_data()
    
    from PaperGraph import *
    p_graph=PaperGraph(name='li_guo')
    p_graph.set_paper_info(data['li_guo'])
    
    
    num=300
    for i in range(num):
        for j in range(num):
            if j<=i:continue
            d=p_graph.dist1(p_graph.get_pid_at(i),p_graph.get_pid_at(j))
            if d==1:
                print(d)