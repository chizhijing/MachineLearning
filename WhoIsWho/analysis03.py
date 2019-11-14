# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:18:30 2019

@author: chizj
"""
from analysis02 import get_train_data,get_valid_data,pairwise_f1
from analysis02 import get_paper_detail,get_graph_by_coauthor,get_coauthor_infor
from analysis02 import nx,copy,json
import nltk
from nltk.corpus import stopwords

my_stopwords="a : “ ” , v the its and as on 's ( ) : % . based".split(' ')
def cal_f1_by_graph(d_graph,tag):
    model_dict={}
    res_dict={}
    for author_select,graph in d_graph.items():
        conn_comp=list(nx.connected_components(graph))
        conn_comp=[list(c) for c in conn_comp]
        model_result={author_select:conn_comp}
        model_dict[author_select]=conn_comp
        real_result={author_select:tag[author_select]}
        f1=pairwise_f1(real_result,model_result)
        res_dict[author_select]=(len(tag[author_select]),len(conn_comp),f1)
        print(author_select,'真实聚类数',len(tag[author_select]),'预测聚类数',len(conn_comp),'%.2f'%f1)         
    return model_dict,res_dict

def title_token(title):
    words=nltk.word_tokenize(title)
    words=[w.lower() for w in words]
    words=[w for w in words if w not in stopwords.words('english')]
    words=[w for w in words if w not in my_stopwords]
    return words

def update_graph(d_graph,v_data):
    graph_copy=copy.deepcopy(d_graph)
    for author,graph in graph_copy.items():
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(v_data[author])
        # 使用keywords信息进行连接
        for index1,k1 in enumerate(keywords):
            k1_stand=[k.lower() for k in k1]
            s1=set(k1_stand)
            for index2,k2 in enumerate(keywords):
                if index2<=index1: continue
                k2_stand=[k.lower() for k in k2]
                s2=set(k2_stand)
                len_set=len(s1&s2)
                if len_set>=1:
                    coauthor_infor=get_coauthor_infor(authors[index1],authors[index2])
                    if coauthor_infor[2]>1: # 单独的keywords不够区分，需要加强合作者进行增强判断
                        graph.add_edge(pids[index1],pids[index2])
    return graph_copy

def update_graph2(d_graph,v_data):
    graph_copy=copy.deepcopy(d_graph)
    for author,graph in graph_copy.items():
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(v_data[author])
        # 使用keywords信息进行连接
        for index1,k1 in enumerate(keywords):
            k1_stand=[k.lower() for k in k1]
            s1=set(k1_stand)
            for index2,k2 in enumerate(keywords):
                if index2<=index1: continue
                k2_stand=[k.lower() for k in k2]
                s2=set(k2_stand)
                len_set=len(s1&s2)
                if len_set>=1:
                    
                    if graph.neighbors(pids[index1])==0 or graph.neighbors(pids[index2])==0: # 
                        graph.add_edge(pids[index1],pids[index2])
    return graph_copy
        
def analysis_author(author,tag,v_data):
    paper_clusters=tag[author]
    pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(v_data[author])
    for i1,paper_cluster in enumerate(paper_clusters):
        if i1>=2:return
        print()
        for j1,pid1 in enumerate(paper_cluster):
            index1=pids.index(pid1)
            k1=[k.lower() for k in keywords[index1]]
            s1=set(k1)
            for j2,pid2 in enumerate(paper_cluster):
                if j2<=j1:continue
                index2=pids.index(pid2)
                k2=[k.lower() for k in keywords[index2]]
                s2=set(k2)
                len_set=len(s1&s2)
                get_coauthor_infor()
                if len_set>=2:
                    print(len_set)
                    
def graph_dict_to_file(graph_dict,file_name):
    model_dict={}
    for author_select,graph in graph_dict.items():
        conn_comp=list(nx.connected_components(graph))
        conn_comp=[list(c) for c in conn_comp]
        model_dict[author_select]=conn_comp
    json.dump(model_dict, open('result/'+file_name+'.json', 'w', encoding='utf-8'), indent=4)                
            
        

if __name__=='__main__':
    valid_data,tag=get_train_data()
#    valid_data=get_valid_data()
    
    author_list_all=list(valid_data.keys())
    author_list=author_list_all[:]

#   根据共同作者计算保留的图结构     
    graph_dict={}
    for author_select in author_list:
#        print(author_select)
        p_list=valid_data[author_select]
        pids,titles,keywords,abstract,authors,venue,year=get_paper_detail(p_list)
        graph_dict[author_select]=get_graph_by_coauthor(pids,authors,author_select)
    
    o_cluster,o_score=cal_f1_by_graph(graph_dict,tag)
    
    graph_dict_k1=update_graph(graph_dict,valid_data) 
    graph_dict_k2=update_graph2(graph_dict,valid_data) 
    
#    graph_dict_to_file(graph_dict_k1,'disambiguate_analysis03')
    
    k1_cluster,k1_score=cal_f1_by_graph(graph_dict_k1,tag)
    k2_cluster,k2_score=cal_f1_by_graph(graph_dict_k2,tag)
    
    for i,author_select in enumerate(author_list):
        print(author_select,o_score[author_select][0],o_score[author_select][1],
              '%.2f'%o_score[author_select][2],k1_score[author_select][1],'%.2f'%k1_score[author_select][2])
        
    