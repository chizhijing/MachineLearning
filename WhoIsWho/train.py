from common import train_author_path,train_pub_path,json,pairwise_f1
from v3 import disambiguate_by_graph
from gensim.utils import tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim import models
from gensim import similarities
import networkx as nx

#my_stopwords=['a','A','the','The']
my_stopwords=set('a the'.split(' ')) # 自定义停用词

def disambiguate_by_dist(author_papers):
    pass

#  获得同名作者所有文章列表的分词结果   
def get_papar_words(author_papers):
    paper_character=[]
    for i,paper in enumerate(author_papers):
        title= [word.lower() for word in tokenize(paper['title'])]
        abstract=[]
        keywords=[]
        text=[]
#        if 'abstract' in paper.keys() and paper['abstract'] is not None:
#            abstract=[word.lower() for word in tokenize(paper['abstract'])]   
        if 'keywords' in paper.keys() and paper['keywords'] is not None:
            keywords=[word.lower() for word in paper['keywords']]
            
        text=title+abstract+keywords # 合并title,keywords,abstract
        text=[word for word in text if (word not in my_stopwords) and (word not in stopwords.words('english'))]
        paper_character.append(text)
    return paper_character

# 根据相似性矩阵构造图模型
def graph_sim_matrix(sim_matrix,corr=0.3):
    graph=nx.Graph()
    pids=list(range(0,len(sim_matrix)))
    graph.add_nodes_from(pids)
    for i in range(0,sim_matrix.shape[0]):
        if i==sim_matrix.shape[0]-1:break
        for j in range(0,sim_matrix.shape[1]):
            if j<=i:continue
            if sim_matrix[i][j]>corr:
                graph.add_edge(i,j)
    
    conn_comp=list(nx.connected_components(graph))
    conn_comp=[list(c) for c in conn_comp]
#    print(conn_comp)
    return conn_comp

# 通过文本相似性的图模型构造的聚类结果
def disambiguate_by_text_sim(validate_data,corr=0.3):
    res_dict={}
    print('不同名作者数',len(validate_data))
    for i,author in enumerate(validate_data.keys()):
        author_papers=validate_data[author]
        if len(author_papers) ==0: 
            res_dict[author]=[]
#        print(i,author,len(author_papers))
        else:
            paper_words=get_papar_words(author_papers)
            dictionary=corpora.Dictionary(paper_words)
            bow_corpus=[dictionary.doc2bow(wl) for wl in paper_words] # 语料向量化
            tfidf=models.TfidfModel(bow_corpus) # 基于向量化的语料构建tfidf模型
            
            index = similarities.Similarity('E:\\gensim_test',tfidf[bow_corpus],len(dictionary))
            sims=index[tfidf[bow_corpus]] # 计算相似性矩阵
            i_cluster=graph_sim_matrix(sims,corr)
            author_cluster=[[author_papers[index]['id'] for index in l_inside ] for l_inside in i_cluster]
#            res_realx={}
#            res_realx[author]=res_real[author]
#            print(author,'pairwise-f1',pairwise_f1(res_realx,{author:author_cluster}))
            print(i,author,'文章数',len(author_papers),'消歧后作者数',len(author_cluster))
            res_dict[author]=author_cluster
    return res_dict




if __name__ == "__main__":
    # 读取测试集数据
    train_author_data = json.load(open(train_author_path, 'r', encoding='utf-8'))
    train_pub_data = json.load(open(train_pub_path, 'r', encoding='utf-8'))

    # 选择部分数据进行测试
    author_list=list(train_author_data.keys())
    author_selects=author_list[0:200]
    
    # 人工标识的分类结果
    res_real={}
    for author in author_selects:
        p_merge=[]
        for plist in train_author_data[author].values():
            p_merge.append(plist)
        res_real[author]=p_merge
            
    papers={}
    for author in author_selects:
        p_merge=[]
        for key,value in train_author_data[author].items():
            for p in value:
#                print(p)
                p_merge.append(train_pub_data[p])
        papers[author]=p_merge
    
    # 根据合作者的图模型进行的聚类(名字预处理的结果会好一些，0.25》0.21)
    res_model2=disambiguate_by_graph(papers)
    pairwise_f1(res_real,res_model2)
    
    # 模型测试
    li_guo=papers['li_guo']
    paper_words=get_papar_words(li_guo)
    dictionary=corpora.Dictionary(paper_words)
    bow_corpus=[dictionary.doc2bow(wl) for wl in paper_words] # 语料向量化
    tfidf=models.TfidfModel(bow_corpus) # 基于向量化的语料构建tfidf模型
    
    index = similarities.Similarity('E:\\gensim_test',tfidf[bow_corpus],len(dictionary))
    sims=index[tfidf[bow_corpus]] # 计算相似性矩阵
    i_cluster=graph_sim_matrix(sims,0.15)
    res_modelx={}
    res_modelx['li_guo']=[[li_guo[index]['id'] for index in l_inside ] for l_inside in i_cluster]
    res_realx={}
    res_realx['li_guo']=res_real['li_guo']
    pairwise_f1(res_realx,res_modelx)

    corpus_tfdif=tfidf[corpus]
    lsi=models.LsiModel(corpus_tfdif,id2word=dictionary,num_topics=10)
    lsi.print_topics(10)
    corpus_lsi=lsi[corpus_tfdif]  
    
    # 封装
    valid={key:papers[key] for key in author_selects} # 取部分测试数据
    res_real_xx={key:res_real[key] for key in author_selects}
    
    res_model_xx=disambiguate_by_text_sim(valid,0.15)
    pairwise_f1(res_real_xx,res_model_xx)

    
    
    
    
    
    
    
    
        
    

    