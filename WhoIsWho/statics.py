lai# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:36:14 2019

@author: chizj
"""
from common import get_train_data

import nltk
from gensim.utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances 

#nltk.download('stopwords')
#nltk.download('punkt')

my_stopwords = ['a', 'one', 'two', 'three', 'four', 'six', 'first', 'second', 'third', 'i', 'h', 'l', 'c', 'via', 'iv']

train_data,train_tag=get_train_data()

# 做所有文章title的语料
paper_t_words = {}
for author, paper_list in train_data.items():
    print(author,len(paper_list))
    titles={}
    for paper in paper_list:
        title = paper['title']
        words = tokenize(title)
        words = [w.lower() for w in words]
        words = [w for w in words if w not in nltk.corpus.stopwords.words('english')]
        words = [w for w in words if w not in my_stopwords]
        titles[paper['id']]=words
    paper_t_words[author] = titles

corpus_title=[]
for author,paper_dict in paper_t_words.items():
    corpus_title.extend(paper_dict.values())
    
# 使用sklearn 词袋模型
corpus_for_cv=[]    
for author,paper_list in train_data.items():
    for paper in paper_list:
        corpus_for_cv.append(paper['title'])
        
count_vec=CountVectorizer() 
res_cv_matrix=count_vec.fit_transform(corpus_for_cv)
pairwise_distances(res_cv_matrix,metric='cosine')

  
    

 




