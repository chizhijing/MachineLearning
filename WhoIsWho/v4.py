# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:01:46 2019

@author: chizj
"""
from common import train_author_path,train_pub_path,valid_pub_data_path,valid_row_data_path,json,pairwise_f1
from v3 import disambiguate_by_graph,num_coauthor_paper
from gensim.utils import tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim import models
from gensim import similarities
import networkx as nx
from train import get_papar_words,graph_sim_matrix
from multiprocessing.pool import Pool

# 根据给定同名作者的文章列表，构造图模型，根据文章的相似性，对图模型进行边的构造
def graph_model(author_paper_list, text_sim=0.15, co_num=4):
    # 同名作者的paper id列表
    paper_id=[paper['id'] for paper in author_paper_list]
    # 构造图模型
    graph=nx.Graph()
    graph.add_nodes_from(paper_id)
    # 有共同作者的聚在一起
    for index1,p1 in enumerate(author_paper_list):
        if index1==len(author_paper_list)-1:break
        for index2,p2 in enumerate(author_paper_list[index1+1:]):
            num_co_au=num_coauthor_paper(p1,p2)
            if num_co_au>=co_num:
                graph.add_edge(p1['id'],p2['id'])
    # 有相同文章主题的聚在一起
    if len(author_paper_list)==0:
        pass
    else:
        paper_words=get_papar_words(author_paper_list)
        dictionary=corpora.Dictionary(paper_words)
        bow_corpus=[dictionary.doc2bow(wl) for wl in paper_words] # 语料向量化
        tfidf=models.TfidfModel(bow_corpus) # 基于向量化的语料构建tfidf模型
        index = similarities.Similarity('E:\\gensim_test',tfidf[bow_corpus],len(dictionary))
        sim_matrix=index[tfidf[bow_corpus]] # 计算相似性矩阵
        # 文章之间的相似度超过给定阈值的建立连接(归为一类)
        for i in range(0,sim_matrix.shape[0]):
            if i==sim_matrix.shape[0]-1:break
            for j in range(0,sim_matrix.shape[1]):
                if j<=i:continue
                if sim_matrix[i][j]>text_sim:
                    graph.add_edge(paper_id[i],paper_id[j])
    # 计算联通子图结果
    conn_comp=list(nx.connected_components(graph))
    conn_comp=[list(c) for c in conn_comp]
    return conn_comp

# 通过文本相似性的图模型构造的聚类结果
def disambiguate_by_graph_model(validate_data,corr=0.15,co_num=3):
    res_dict={} # 存放聚类结果
    print('不同名作者数',len(validate_data),'corr',corr,'co_num',co_num)
    for i,author in enumerate(validate_data.keys()):
        author_papers=validate_data[author]
        author_cluster=graph_model(author_papers,corr,co_num)
        print(i,author,'文章数',len(author_papers),'消歧后作者数',len(author_cluster))
        res_dict[author]=author_cluster
    return res_dict


if __name__=='__main__':
    # 验证集结果生成
    #  validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
    #  validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
    #  merge_data = {}
    #  for author in validate_data:
    #      validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]
    #
    #  res=disambiguate_by_graph_model(validate_data,0.15)
    #  json.dump(res, open('result/disambiguate_by_text_sim_1.json', 'w', encoding='utf-8'), indent=4)
    #
     # 在测试集上的验证
     train_author_data = json.load(open(train_author_path, 'r', encoding='utf-8'))
     train_pub_data = json.load(open(train_pub_path, 'r', encoding='utf-8'))

     author_list=list(train_author_data.keys())
     author_selects=author_list[50:100] # 选择部分数据进行测试
     
     train_data={}
     for author in author_selects:
         pid_list=[train_pub_data[p] for pp in train_author_data[author].values() for p in pp]
         train_data[author]=pid_list
     
     real_result={key:[pid for pid in train_author_data[key].values()] for key in author_selects} # 训练数据真实标签
     
     modle_result=disambiguate_by_graph_model(train_data,0.5,2)
     pairwise_f1(real_result,modle_result)
     
     # 采用多进程加快速度
     train_data_list=[]
     for i in range(6):
         train_data={}
         for author in author_list[i:20:6]:
             pid_list=[train_pub_data[p] for pp in train_author_data[author].values() for p in pp]
             train_data[author]=pid_list
         train_data_list.append(train_data)
     pool=Pool(processes=6)
     map_res=pool.map(disambiguate_by_graph_model,train_data_list)
     result_model={}
     for d in map_res:
         result_model.update(d)
     real_result = {key: [pid for pid in train_author_data[key].values()] for key in result_model.keys()}  # 训练数据真实标签
     print(pairwise_f1(real_result,result_model))

    
