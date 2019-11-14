# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:29:49 2019

@author: chizj
"""

from common import get_valid_data,get_train_data,pairwise_f1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from multiprocessing import Pool
from functools import partial
import pandas as pd


def get_title_corpus(data):
    titles=[]
    for author,p_list in data.items():
        for paper in p_list:
            titles.append(paper['title'])
    return titles

# 提取文章列表的信息 ID, content（title+abstract）
def get_text_corpus(paper_list):
    text=[]
    pids=[]
    for paper in paper_list:
        title=paper['title']
        abstract=paper['abstract'] if 'abstract' in paper.keys() else ''
        if abstract is None: abstract=''
#        text.append(title+' '+abstract)
        text.append(title)
        pids.append(paper['id'])
    return pids,text

# 对某一个同名作者的文章进行聚类
def cluster_one_author(author_paper,n_component=100,eps_cluster=0.8):
    paper_ids,text=get_text_corpus(author_paper) # 获取文章id列表，和文章相关信息
    tfidf_model=TfidfVectorizer() # 使用sklearn 构建tfidf模型
    X=tfidf_model.fit_transform(text) # 拟合模型
#    print(X.shape)
    svd = TruncatedSVD(n_component) # SVD降维
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer) # 构建LSA模型
    X = lsa.fit_transform(X) 
#    print(X.shape)
#    explained_variance = svd.explained_variance_ratio_.sum()
    # print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    
    cluster_dbs=DBSCAN(eps=eps_cluster,min_samples=1) # 使用DBSCAN聚类方法
    cluster_dbs.fit(X)
    
    labels=cluster_dbs.labels_ # 样本标签
    l_set=set(labels) # 标签集合
    
    paper_cluster=[]
    paper_id_arr=np.array(paper_ids)
    for c in l_set:
        paper_cluster.append(list(paper_id_arr[labels==c]))
    return paper_cluster
    
#valid_data=get_valid_data()

#titles=get_title_corpus({'heng_li':valid_data['heng_li']})

# j

def cal_train_f1(eps_cluster,n_component,valid_data,tag):
    author_name_list = list(valid_data.keys())
    res_model_all = {}
    res_real_all = {}
    for i, author_name in enumerate(author_name_list):
        author_paper = valid_data[author_name]  # 同名作者文章列表
        if len(author_paper) == 0:
            print(author_name, '没有文章')
            continue
        res_model_author = {author_name: cluster_one_author(author_paper, 10, 0.5)}
        res_real_author = {author_name: tag[author_name]}
        # f1 = pairwise_f1(res_real_author, res_model_author)
        # print('i=', i, '同名作者:', author_name, '文章数量:', len(author_paper), '预测消歧作者数', len(res_model_author[author_name]),
        #       '实际消歧作者数', len(tag[author_name]), 'f1-value', f1)
        res_model_all.update(res_model_author)
        res_real_all.update(res_real_author)
    f1=pairwise_f1(res_real_all, res_model_all)
    return n_component,eps_cluster,f1

def new_cal(t_eps_n,valid_data,tag):
    res = cal_train_f1(t_eps_n[0],t_eps_n[1],valid_data,tag)
    print(t_eps_n[0],t_eps_n[1],res)
    return res

if __name__=='__main__':
    valid_data, tag = get_train_data()

    authors=list(valid_data.keys())
    # sample_data={}
    # sample_tag={}
    # for author in authors[0:3]:
    #     sample_data[author]=valid_data[author]
    #     sample_tag[author]=tag[author]

    eps_list=[0.05+i*0.1for i in range(20)]
    n_com=range(3,100,10)
    tuple_eps_ncom_list=[(eps,n) for eps in eps_list for n in n_com]
    pool=Pool(4)
    print('map begin')
    map_res=pool.map(partial(new_cal,valid_data=valid_data,tag=tag),tuple_eps_ncom_list)
    # map_res = pool.map(partial(cal_train_f1, valid_data=sample_data, tag=sample_tag, n_component=10), eps_list)
    pd.DataFrame(map_res).to_csv('.\\map_result.csv')
    print(map_res)






