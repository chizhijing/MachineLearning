from graph import get_train_data
from collections import defaultdict
import nltk
import re
import random
from gensim import corpora
from gensim import models
from gensim import similarities

if __name__ == '__main__':
    train_data,train_tag=get_train_data()

    titles=[]
    for author,papers in train_data.items():
        for pid,p_info in papers.items():
            titles.append(p_info['title'])

    my_stop_list=set('for a an of the and to in on by as at from with under its some based & ]- ≠ < > . : , ( ) i ii iii iv v i 1 2 3 4 5 6 7 8'.split(' '))

#    titles=[re.sub(r"\s*-\s*", "-", title) for title in titles]
    titles=[re.sub(r"\s*-\s*", " ", title) for title in titles]

    # title_words=[[word for word in title.lower().split() if word not in my_stop_list] for title in titles]
    title_words=[[w for w in nltk.word_tokenize(title.lower()) if w not in my_stop_list] for title in titles]

    # 计算词频
    freq=defaultdict(int)
    for title in title_words:
        for w in title:
            freq[w] +=1
    
    for w in freq.keys():
        if freq[w]>5000:
            print(w,freq[w])

    process_tw=[[w for w in title if freq[w]>1] for title in title_words] # 删除只出现一次的词语
    
    # 构建字典
    dictionary=corpora.Dictionary(process_tw)
    bow_corpus=[dictionary.doc2bow(tw) for tw in process_tw] # 语料向量化

    # tfidf模型
    tfidf=models.TfidfModel(bow_corpus)
    print(tfidf[bow_corpus[0]])
    tfidf_corpus = [tfidf[bc] for bc in bow_corpus]

    # lsi模型
    lsi=models.LsiModel(tfidf_corpus,id2word=dictionary,num_topics=4)
    lsi_corpus=[lsi[bc] for bc in tfidf_corpus]
    
    similarity_lsi=similarities.MatrixSimilarity(lsi_corpus[0:10])
    similarity_lsi[lsi_corpus[0:10]]
    print(lsi_corpus[0])
    

