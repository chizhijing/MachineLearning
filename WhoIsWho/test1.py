# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:00:33 2019

@author: chizj
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim import models

import re
# 数据预处理

# 预处理名字
def precessname(name):   
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name) 
    return name

# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        org = org.split(';')[0]  # 多个机构只取第一个
    return org

#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content

def get_org(co_authors, author_name):
    for au in co_authors:
        name = precessname(au['name'])
        name = name.split('_')
        if ('_'.join(name) == author_name or '_'.join(name[::-1]) == author_name) and 'org' in au:
            return au['org']
    return ''


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 训练集分析
train_row_data_path = 'data/train/train_author.json'
train_pub_data_path = 'data/train/train_pub.json'

train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
train_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))

author_paper={}
for author_name in train_data.keys():
    p_list=[]
    for author_id in train_data[author_name].keys():
        p_list.extend(train_data[author_name][author_id])
    author_paper[author_name]=p_list

# 对某个姓名的所有文章进行分类
author_select='hui_wang'     
print('姓名：',author_select,"文章数:",len(author_paper[author_select]),'真实作者数:',len(train_data[author_select].keys()))  
abstracts=[train_pub_data[p_id]['abstract'] for p_id in author_paper[author_select]]

words=[nltk.word_tokenize(abstract) for abstract in abstracts] # 分词
words_drop_stop=[[w for w in word if w not in stopwords.words('english')] for word in words] # 停用词

dictionary=Dictionary(words_drop_stop)
vector=[dictionary.doc2bow(doc) for doc in words_drop_stop] #语料库
print('词汇量：',len(dictionary.token2id))

tfidf=models.TfidfModel(vector)
tfidf_vectors = tfidf[vector]

print(len(tfidf_vectors))
print(len(tfidf_vectors[0]))