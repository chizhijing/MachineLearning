import json
import networkx as nx
from common import get_coauthor_infor,get_author_org,get_org_infor,pairwise_f1
import re
import nltk
from collections import defaultdict
from gensim import corpora,models,similarities
import copy

valid_row_data_path = 'data/sna_data/sna_valid_author_raw.json'
valid_pub_data_path = 'data/sna_data/sna_valid_pub.json'
train_author_path = 'data/train/train_author.json'
train_pub_path = 'data/train/train_pub.json'

my_stopwords=set('for a an of the and to in on by as at from with under its some based '
                 '& ]- ≠ < > . : , ( ) i ii iii iv v i 1 2 3 4 5 6 7 8'.split(' '))

def get_train_data(author_list=None):
    train_author_data = json.load(open(train_author_path, 'r', encoding='utf-8'))
    train_pub_data = json.load(open(train_pub_path, 'r', encoding='utf-8'))
    train_data = {}
    train_tag = {}
    if author_list is None:
        author_list = list(train_author_data.keys())

    for author in author_list:
        pid_dict={}
        for real_aid,plist in train_author_data[author].items():
            for pid in plist:
                pid_dict[pid] = train_pub_data[pid]
        train_data[author] = pid_dict
        train_tag[author] = list(train_author_data[author].values())
    return train_data, train_tag

class GraphBase:
    def __init__(self,nodes_info_dict):
        self.nodes=list(nodes_info_dict.keys())
        self.nodes_info_dict=nodes_info_dict
        self.graph=self.init_graph()

    def init_graph(self):
        graph=nx.Graph()
        graph.add_nodes_from(self.nodes)
        return graph

    def set_init_graph(self,init_graph):
        self.graph=copy.deepcopy(init_graph)

    def add_edge(self):
        pass

    def get_connected_components(self):
        conn_comp = list(nx.connected_components(self.graph))
        conn_comp = [list(c) for c in conn_comp]
        return conn_comp

class GraphAuthors(GraphBase):
    def __init__(self,nodes_info_dict,author_name):
        super().__init__(nodes_info_dict)
        self.author_name=author_name

    def add_edge(self):
        for i1,node1 in enumerate(self.nodes):
            paper1=self.nodes_info_dict[node1]
            if 'authors' in paper1.keys() and paper1['authors'] is not None:
                coauthor1=paper1['authors']
            else:
                coauthor1=''
            org1 = get_author_org(coauthor1, self.author_name)
            for i2,node2 in enumerate(self.nodes):
                if i2<=i1:continue
                paper2=self.nodes_info_dict[node2]
                if 'authors' in paper2.keys() and paper2['authors'] is not None:
                    coauthor2 = paper2['authors']
                else:
                    coauthor2 = ''
                # 使用共同作者进行的判断
                len_a1, len_a2, len_a1_a2 = get_coauthor_infor(coauthor1, coauthor2)
                if (len_a1_a2 > 2) or (len_a1_a2 == 2 and max(len_a1, len_a2) <= 4):
                    self.graph.add_edge(node1, node2)
                    continue
                # 使用组织机构进行的判断
                org2 = get_author_org(coauthor2, self.author_name)
                len_o1, len_o2, dist_o1_o2 = get_org_infor(org1, org2)
                if len_o1 == 0 or len_o2 == 0: continue  # 组织机构为空的不能进行判断
                if dist_o1_o2 == 0 or dist_o1_o2 < min(len_o1, len_o2) / 5:
                    self.graph.add_edge(node1, node2)
                    continue
    def add_edge2(self):
        for i1,node1 in enumerate(self.nodes):
            paper1=self.nodes_info_dict[node1]
            if 'authors' in paper1.keys() and paper1['authors'] is not None:
                coauthor1=paper1['authors']
            else:
                coauthor1=''
            org1 = get_author_org(coauthor1, self.author_name)
            for i2,node2 in enumerate(self.nodes):
                if i2<=i1:continue
                paper2=self.nodes_info_dict[node2]
                if 'authors' in paper2.keys() and paper2['authors'] is not None:
                    coauthor2 = paper2['authors']
                else:
                    coauthor2 = ''
                # 使用共同作者进行的判断
                len_a1, len_a2, len_a1_a2 = get_coauthor_infor(coauthor1, coauthor2)
                if len_a1_a2 > 3:
                    self.graph.add_edge(node1, node2)
                    continue
                # 使用组织机构进行的判断
                org2 = get_author_org(coauthor2, self.author_name)
                len_o1, len_o2, dist_o1_o2 = get_org_infor(org1, org2)
                if len_o1 == 0 or len_o2 == 0: continue  # 组织机构为空的不能进行判断
                if dist_o1_o2 == 0:
                    self.graph.add_edge(node1, node2)
                    continue

class GraphClusters(GraphBase):
    def __init__(self,nodes_info_dict):
        super().__init__(nodes_info_dict)

class GraphTitles(GraphBase):
    def __init__(self,nodes_info_dict,eps_dist=0.9):
        super().__init__(nodes_info_dict)
        self.sim_matrix=None
        self.eps_dist=eps_dist

    def add_edge(self):
        if self.sim_matrix is None: return
        for i1, node1 in enumerate(self.nodes):
            for i2, node2 in enumerate(self.nodes):
                if i2 <= i1: continue
                if self.sim_matrix[i1][i2]>self.eps_dist:
                    self.graph.add_edge(node1,node2)

    def get_dist_matrix(self):
        titles=[paper_info['title'] for pid,paper_info in self.nodes_info_dict.items()]
        titles = [re.sub(r"\s*-\s*", " ", title) for title in titles]
        title_words = [[w for w in nltk.word_tokenize(title.lower()) if w not in my_stopwords] for title in titles]
        # 计算词频
        freq = defaultdict(int)
        for title in title_words:
            for w in title:
                freq[w] += 1
        process_tw = [[w for w in title if freq[w] > 1] for title in title_words]  # 删除只出现一次的词语
        # 构建字典
        dictionary = corpora.Dictionary(process_tw)
        bow_corpus = [dictionary.doc2bow(tw) for tw in process_tw]  # 语料向量化
        # tfidf模型
        tfidf = models.TfidfModel(bow_corpus)
        tfidf_corpus = [tfidf[bc] for bc in bow_corpus]
        # lsi模型
        lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=4)
        lsi_corpus = [lsi[bc] for bc in tfidf_corpus]
        similarity_lsi = similarities.MatrixSimilarity(lsi_corpus)
        self.sim_matrix=similarity_lsi[lsi_corpus]

    def get_cluster(self):
        init_cluster=self.get_connected_components()
        
        

    def dist_two_paper_sets(self,pid1s,pid2s):
        max_dist=-100
        for i1,p1 in enumerate(pid1s):
            index1=self.nodes.index(p1)
            min_dist=10000
            for i2,p2 in enumerate(pid2s):
                index2=self.nodes.index(p2)
                if self.sim_matrix[index1][index2]<min_dist:
                    min_dist=self.sim_matrix[index1][index2]
            if min_dist>max_dist:
                max_dist=min_dist
                

if __name__=="__main__":
    author_list=['li_guo', 'bo_shen', 'di_wang', 'long_wang', 'qiang_xu', 
                 'xiang_wang', 'changming_liu', 'kenji_kaneko', 'guohua_chen', 'hai_jin', 
                 'jia_li', 'guoliang_li', 'lan_wang', 'alessandro_giuliani', 'jiang_he', 
                 'xiang_gao', 'jianping_wu', 'peng_shi', 'feng_wu', 'jing_zhu']
    t_data,t_tag=get_train_data(author_list)

    
    graph_author_dict={}
    graph_author_dict2={}
    graph_title_dict={}
    graph_at_dict={}
    graph_a2t_dict={}
    
    # 使用作者信息进行聚类的结果
    for author_select in author_list:
        graph_a=GraphAuthors(t_data[author_select],author_select)
        graph_a.add_edge()
        graph_author_dict[author_select]=graph_a
    
    # 使用作者信息2进行聚类的结果
    for author_select in author_list:
        graph_a2=GraphAuthors(t_data[author_select],author_select)
        graph_a2.add_edge2()
        graph_author_dict2[author_select]=graph_a2

    # 使用Title进行聚类的结果    
    for author_select in author_list:
        graph_t=GraphTitles(t_data[author_select],0.995)
        graph_t.get_dist_matrix()
        graph_t.add_edge()
        graph_title_dict[author_select]=graph_t

    # 使用作者信息+title进行聚类
    for author_select in author_list:
        graph_at=GraphTitles(t_data[author_select],0.9999)
        graph_at.set_init_graph(graph_author_dict[author_select].graph)
        graph_at.get_dist_matrix()
        graph_at.add_edge()
        graph_at_dict[author_select]=graph_at
    
    # 使用作者信息2+title进行聚类
    for author_select in author_list:
        graph_a2t=GraphTitles(t_data[author_select],0.99)
        graph_a2t.set_init_graph(graph_author_dict2[author_select].graph)
        graph_a2t.get_dist_matrix()
        graph_a2t.add_edge()
        graph_a2t_dict[author_select]=graph_a2t
        
    
    for author_select in author_list:
        real_res={author_select:t_tag[author_select]}
        model_author={author_select: graph_author_dict[author_select].get_connected_components()}
        model_author2={author_select: graph_author_dict2[author_select].get_connected_components()}
        model_title={author_select:graph_title_dict[author_select].get_connected_components()}
        model_at={author_select:graph_at_dict[author_select].get_connected_components()}
        model_a2t={author_select:graph_a2t_dict[author_select].get_connected_components()}
        print(author_select,len(real_res[author_select]))
        print('----> model_author',
              '%.2f'%pairwise_f1(real_res,model_author),len(model_author[author_select]))
        print('----> model_author2',
              '%.2f'%pairwise_f1(real_res,model_author2),len(model_author2[author_select]))
        print('----> model_title',
              '%.2f'%pairwise_f1(real_res,model_title),len(model_title[author_select]))
        print('----> model_at',
              '%.2f'%pairwise_f1(real_res,model_at),len(model_at[author_select]))
        print('----> model_a2t',
              '%.2f'%pairwise_f1(real_res,model_a2t),len(model_a2t[author_select]))




