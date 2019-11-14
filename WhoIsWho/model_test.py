# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:54:20 2019

@author: chizj
"""

from common import *

validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in validate_data:
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]
    
title=[validate_pub_data[pid]['title'] for pid in validate_pub_data.keys()]
abstract=[value['abstract'] if 'abstract' in value.keys()  else " " for value in validate_pub_data.values()]
keywords=[value['keywords'] if 'keywords' in value.keys()  else " " for value in validate_pub_data.values()]

from gensim.utils import tokenize
t_title=[list(tokenize(tt)) for tt in title] # 分词

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
my_stopwords=['a','A','the','The']
[t_list.remove(t) for t_list in t_title for t in t_list if t in my_stopwords]
[t_list.remove(t) for t_list in t_title for t in t_list if t in stopwords.words('english')]

from gensim import corpora
dictionary=corpora.Dictionary(t_title)
corpus=[dictionary.doc2bow(tt) for tt in t_title] # 语料库制作

from gensim import models
tfidf=models.TfidfModel(corpus)
corpus_tfdif=tfidf[corpus]

lsi=models.LsiModel(corpus_tfdif,id2word=dictionary,num_topics=10)
lsi.print_topics(10)

corpus_lsi=lsi[corpus_tfdif]

from gensim import similarities
index=similarities.MatrixSimilarity(corpus_lsi)

sims=index[corpus_lsi[0]]
print(sims[1])
