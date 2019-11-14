# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:38:05 2019

@author: chizj
"""

from graph import get_train_data
from graph import GraphAuthors
from common import pairwise_f1

if __name__=='__main__':
    a_list=['li_guo','qiang_xu']
    train_data,train_tag=get_train_data(a_list)

    g_author=GraphAuthors(train_data['li_guo'],'li_guo')
    g_author.add_edge()
    model_res={'li_guo':g_author.get_connected_components()}
    real_res={'li_guo':train_tag['li_guo']}

    pairwise_f1(real_res,model_res)
