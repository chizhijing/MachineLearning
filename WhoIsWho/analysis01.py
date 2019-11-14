# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:21:36 2019

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

# 计算两个共同作者列表的相同作者数目(不包括当前消歧的作者)
def get_coauthor_num(co_list1,co_list2,author_name):
    counter=0
    for c1 in co_list1:
        name1=precessname(c1['name'])
        if name1==author_name:continue
        for c2 in co_list2:
            name2=precessname(c2['name'])
            if name2==name1:
                counter+=1
    return counter

#  计算两个共同作者列表的相同作者数目(不包括当前消歧的作者),效果好于get_coauthor_num
def get_coauthor_num2(co_list1,co_list2):
    name1_list=[precessname(c1['name']) for c1 in co_list1]
    name2_list=[precessname(c2['name']) for c2 in co_list2]
    return len(set(name1_list) & set(name2_list))

# 计算两个共同作者列表的相同作者比例,某些合作者很多，合作者同名的可能性变大，使用比例可以降低
def get_coauthor_ratio(co_list1,co_list2):
    name1_list=[precessname(c1['name']) for c1 in co_list1]
    name2_list=[precessname(c2['name']) for c2 in co_list2]
    if len(name1_list)-1<=0 or len(name2_list)-1<=0: return 0
    return (len(set(name1_list) & set(name2_list))-1)/max(len(name1_list)-1,len(name2_list)-1)

def get_coauthor_infor(co_list1,co_list2):
    name1_list=[precessname(c1['name']) for c1 in co_list1]
    name2_list=[precessname(c2['name']) for c2 in co_list2]
    return (len(name1_list),len(name2_list),len(set(name1_list) & set(name2_list)))
def get_org_infor(org1,org2):
    p_org1=preprocessorg(org1)
    p_org2=preprocessorg(org2)
    return (len(p_org1),len(p_org2),lv.distance(p_org1,p_org2))
def get_keywords_infor(k1_list,k2_list):
    return (len(k1_list),len(k2_list),len(set(k1_list)& set(k2_list)))

# 判断给定的两个组织是否为同一个组织
def IsSameOrg(org1,org2,cal_type=1):
    if org1=='' or org2=='': return False
    if cal_type==1:
        if org1==org2: return True
        return False
    if cal_type==2:
        if lv.distance(preprocessorg(org1),preprocessorg(org2))<5: return True
        return False
    return False

# 随机产生聚类，每篇文章一个作者
def rand_cluster(paper_list):
    res_result=[]
    if len(paper_list) % 2 ==0 :
        for i in range(0,len(paper_list),2):
            res_result.append([paper_list[i]['id'],paper_list[i+1]['id']])
    else:
        for i in range(0,len(paper_list)-1,2):
            res_result.append([paper_list[i]['id'],paper_list[i+1]['id']])
        res_result[-1].extend(paper_list[-1]['id'])
    return res_result

# 根据共同作者建立的图数据结构
def get_graph_by_coauthor(pids,authors,author_name):
     # 构建图数据结构
    graph=nx.Graph()
    graph.add_nodes_from(pids)
                
    # 具有共同作者的进行聚类                
    for index1,coauthor1 in enumerate(authors):
        if index1==len(authors)-1: break
        for index2,coauthor2 in enumerate(authors):
            if index2<=index1:continue
#            num=get_coauthor_num(coauthor1,coauthor2,author_name)
#            num = get_coauthor_num2(coauthor1,coauthor2)-1
#            if num>=1:
#                graph.add_edge(pids[index1],pids[index2])
            ratio = get_coauthor_ratio(coauthor1,coauthor2)
            if ratio>0.3:
               graph.add_edge(pids[index1],pids[index2]) 
    return graph

# 根据已有的图结构，利用org信息更新新的图结构
def get_new_graph_by_org(graph,pids,authors,author_name):
    graph_copy=copy.deepcopy(graph)
    for index1,coauthor1 in enumerate(authors):
        if index1==len(authors)-1: break
        org1=get_author_org(coauthor1,author_name)
        if org1=='':continue
        for index2,coauthor2 in enumerate(authors):
            if index2<=index1:continue
            org2=get_author_org(coauthor2,author_name)
            if org2=='':continue
        #            if IsSameOrg(org1,org2,cal_type=1): 
            if IsSameOrg(org1,org2,cal_type=2): 
                graph_copy.add_edge(pids[index1],pids[index2])
    return graph_copy


# 强规则聚类
def co_author_cluster(pids,authors,author_name):
    # 构建图数据结构
    graph=nx.Graph()
    graph.add_nodes_from(pids)
    
    # 使用预处理后的组织名字完全一样(待优化)
    for index1,coauthor1 in enumerate(authors):
        if index1==len(authors)-1: break
        org1=get_author_org(coauthor1,author_name)
        if org1=='':continue
        for index2,coauthor2 in enumerate(authors):
            if index2<=index1:continue
            org2=get_author_org(coauthor2,author_name)
            if org2=='':continue
#            if IsSameOrg(org1,org2,cal_type=1): 
            if IsSameOrg(org1,org2,cal_type=2): 
                graph.add_edge(pids[index1],pids[index2])
                
    # 具有共同作者的进行聚类                
#    for index1,coauthor1 in enumerate(authors):
#        if index1==len(authors)-1: break
#        for index2,coauthor2 in enumerate(authors):
#            if index2<=index1:continue
##            num=get_coauthor_num(coauthor1,coauthor2,author_name)
#            num = get_coauthor_num2(coauthor1,coauthor2)-1
#            if num>=1:
#                graph.add_edge(pids[index1],pids[index2])
    
    
    conn_comp=list(nx.connected_components(graph))
    conn_comp=[list(c) for c in conn_comp]   
    return conn_comp      

            



if __name__=='__main__':
    
    valid_data, tag = get_train_data()
#    valid_data = get_valid_data()
    f1_dict={}
    
    # 分析某个同名作者的文章信息
    author_list_all=list(tag.keys())
    author_list=author_list_all[1:20]
    for author_select in author_list:
        p_list=valid_data[author_select]
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
        
        real_result={author_select:tag[author_select]}
#        model_result={author_select:rand_cluster(p_list)}
        model_result={author_select:co_author_cluster(pids,authors,author_select)}
        f1=pairwise_f1(real_result,model_result)
        print(author_select,f1)
        f1_dict[author_select]=f1
            
#    org_test=[]
#    for co_au in authors:
#        for au in co_au:
#            if precessname(au['name']) == 'bo_shen':
#                oo=au['org'].split(';')[0]
#                if oo!='':org_test.append(oo)
        
    author_list_all=list(valid_data.keys())
    author_list=author_list_all[0:20]
#   根据共同作者计算保留的图结构     
    graph_dict={}
    for author_select in author_list:
#        print(author_select)
        p_list=valid_data[author_select]
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
        graph_dict[author_select]=get_graph_by_coauthor(pids,authors,author_select)

#  使用已有的图根据org更新的图结构    
    new_graph_dict={}
    for author_select in graph_dict.keys():
        p_list=valid_data[author_select]
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
        new_graph_dict[author_select]=get_new_graph_by_org(graph_dict[author_select],pids,authors,author_select)
        
# 聚类情况分析(在训练集的情况下的比较)
    model_dict={}
    for author_select,graph in new_graph_dict.items():
        conn_comp=list(nx.connected_components(graph))
        conn_comp=[list(c) for c in conn_comp]
        model_result={author_select:conn_comp}
        model_dict[author_select]=conn_comp
        real_result={author_select:tag[author_select]}
        f1=pairwise_f1(real_result,model_result)
        print(author_select,'真实聚类数',len(tag[author_select]),'预测聚类数',len(conn_comp),'%.2f'%f1)
    
    json.dump(model_dict, open('result/disambiguate_analysis01.json', 'w', encoding='utf-8'), indent=4)
###########################################################################################
# 某个同名作者的文章信息查看
    author_select='xiang_gao'
    p_list=valid_data[author_select]
    pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
    
    # 具有共同作者
    info_list=[]               
    for index1,coauthor1 in enumerate(authors):
        if index1==len(authors)-1: break
        for index2,coauthor2 in enumerate(authors):
            if index2<=index1:continue
            num = get_coauthor_num2(coauthor1,coauthor2)-1
            if num>=1:
                info_list.append((index1,index2,num,len(coauthor1),len(coauthor2)))
   
    i=2
    print(info_list[i])
    print(authors[info_list[i][0]])
    print(authors[info_list[i][1]])
    
    