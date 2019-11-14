# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:34:10 2019

@author: chizj
"""

from common import pairwise_f1,precessname,preprocessorg
from common import get_valid_data,get_train_data
import Levenshtein as lv
import networkx as nx
import copy
import json

# 提取文章列表的信息 ID, title,abstract,keywords
def get_paper_detail(paper_list):
    titles=[]
    abstracts=[]
    keywords=[]
    pids=[]
    author_list=[]
    venues=[]
    years=[]
    for paper in paper_list:
        title=paper['title']
        abstract=paper['abstract'] if 'abstract' in paper.keys() else ''
        if abstract is None: abstract=''
        keyword=paper['keywords'] if 'keywords' in paper.keys() else ''
        if keyword is None: keyword=''
        authors=paper['authors'] if 'authors' in paper.keys() else ''
        if authors is None: authors=''
        venue=paper['venue'] if 'venue' in paper.keys() else ''
        if venue is None: venue=''
        year=paper['year'] if 'year' in paper.keys() else ''
        if year is None: year=''
        
        titles.append(title)
        abstracts.append(abstract)
        keywords.append(keyword)
        pids.append(paper['id'])
        author_list.append(authors)
        venues.append(venue)
        years.append(year)
    return pids,titles,keywords,abstracts,author_list,venues,years

# 返回给定共同作者列表中，需要消歧作者的组织机构
def get_author_org(co_list,author_name):
    for co_author in co_list:
        if precessname(co_author['name'])==author_name:
            if 'org' in co_author.keys():
                return co_author['org']
            else:
                return ''
    return ''

def get_coauthor_infor(co_list1,co_list2):
    name1_list=[precessname(c1['name']) for c1 in co_list1]
    name2_list=[precessname(c2['name']) for c2 in co_list2]
    return len(name1_list),len(name2_list),len(set(name1_list) & set(name2_list))

def get_org_infor(org1,org2):
    p_org1=preprocessorg(org1)
    p_org2=preprocessorg(org2)
    return len(p_org1),len(p_org2),lv.distance(p_org1,p_org2)

def get_keywords_infor(k1_list,k2_list):
    return len(k1_list),len(k2_list),len(set(k1_list)& set(k2_list))

# 根据共同作者建立的图数据结构
def get_graph_by_coauthor(pids,authors,author_name):
     # 构建图数据结构
    graph=nx.Graph()
    graph.add_nodes_from(pids)
                
    # 具有共同作者的进行聚类                
    for index1,coauthor1 in enumerate(authors):
        if index1==len(authors)-1: break
        org1=get_author_org(coauthor1,author_name)
        for index2,coauthor2 in enumerate(authors):
            if index2<=index1:continue
            # 使用共同作者进行的判断
            len_a1,len_a2,len_a1_a2=get_coauthor_infor(coauthor1,coauthor2)
            if len_a1_a2>2:
               graph.add_edge(pids[index1],pids[index2]) 
               continue
            else:
               if len_a1_a2==2 and max(len_a1,len_a2)<=4:
                  graph.add_edge(pids[index1],pids[index2]) 
                  continue
            # 使用组织机构进行的判断
            org2=get_author_org(coauthor2,author_name)
            len_o1,len_o2,dist_o1_o2=get_org_infor(org1,org2)
            if len_o1==0 or len_o2==0:continue # 组织机构为空的不能进行判断
            if dist_o1_o2==0:
                graph.add_edge(pids[index1],pids[index2]) 
                continue
            if dist_o1_o2<min(len_o1,len_o2)/5:
                graph.add_edge(pids[index1],pids[index2]) 
                continue
    return graph

def cal_f1_each_author(model_res,real_res):
    for author in model_res.keys():
        model_author={author:model_res[author]}
        real_author={author:real_res[author]}
        f1=pairwise_f1(real_author,model_author)
        print(author,'真实聚类数',len(real_author[author]),'预测聚类数',len(model_author[author]),'%.2f'%f1)


if __name__=='__main__':
    
    valid_data, tag = get_train_data()
#    valid_data = get_valid_data()
            
    author_list_all=list(valid_data.keys())
    author_list=author_list_all[:]
#   根据共同作者计算保留的图结构     
    graph_dict={}
    for author_select in author_list:
#        print(author_select)
        p_list=valid_data[author_select]
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
        graph_dict[author_select]=get_graph_by_coauthor(pids,authors,author_select)
        
#    json.dump(model_dict, open('result/disambiguate_analysis02.json', 'w', encoding='utf-8'), indent=4)
    
#  使用已有的图根据org更新的图结构    
#    new_graph_dict={}
#    for author_select in graph_dict.keys():
#        p_list=valid_data[author_select]
#        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
#        new_graph_dict[author_select]=get_new_graph_by_org(graph_dict[author_select],pids,authors,author_select)
        
# 聚类情况分析(在训练集的情况下的比较)
    model_dict={}
    for author_select,graph in graph_dict.items():
        conn_comp=list(nx.connected_components(graph))
        conn_comp=[list(c) for c in conn_comp]
        model_result={author_select:conn_comp}
        model_dict[author_select]=conn_comp
        real_result={author_select:tag[author_select]}
        f1=pairwise_f1(real_result,model_result)
        print(author_select,'真实聚类数',len(tag[author_select]),'预测聚类数',len(conn_comp),'%.2f'%f1)
    
#    json.dump(model_dict, open('result/disambiguate_analysis02.json', 'w', encoding='utf-8'), indent=4)
###########################################################################################
# 某个同名作者的文章信息查看
    author_select='xiang_gao'
    p_list=valid_data[author_select]
    pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
    
    # 查看真实的同一作者的文章情况
    pids_real_cluster=tag[author_select]
    pids_cluster_select=pids_real_cluster[1]
        
    for i1,pid in enumerate(pids_real_cluster[0]):
        index1=pids.index(pid)
        author_list1=authors[index1]
        for i2,pid2 in enumerate(pids_real_cluster[1]):
            index2=pids.index(pid2)
            author_list2=authors[index2]
            len1,len2,len3=get_coauthor_infor(author_list1,author_list2)
            if len3>2: 
                print(pids[index1],pids[index2],len1,len2,len3)

    