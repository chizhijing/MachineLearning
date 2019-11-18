import json
import networkx as nx
from common import get_coauthor_infor,get_author_org,get_org_infor,pairwise_f1,preprocessorg,process_org2,precessname
import re
import nltk
from collections import defaultdict
from gensim import corpora,models,similarities
import copy
import itertools

valid_row_data_path = 'data/sna_data/sna_valid_author_raw.json'
valid_pub_data_path = 'data/sna_data/sna_valid_pub.json'
train_author_path = 'data/train/train_author.json'
train_pub_path = 'data/train/train_pub.json'

my_stopwords=set('for a an of the and to in on by as at from with under its some based '
                 '& ]- ≠ < > . : , ( ) i ii iii iv v i 1 2 3 4 5 6 7 8'.split(' '))


def is_same_org(org1, org2):
    p_org1=process_org2(org1)
    p_org2 = process_org2(org2)
    print(org1,p_org1)
    print(org1,p_org2)

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
        self.author_org=None
        self.coauthors=None
        self.get_authors_org()

    def get_authors_org(self):
        self.author_org=[]
        self.coauthors=[]
        for pid,paper_info in self.nodes_info_dict.items():
            if 'authors' in paper_info.keys() and paper_info['authors'] is not None:
                coauthor=paper_info['authors']
            else:
                coauthor=''
            org = get_author_org(coauthor, self.author_name)
            org=preprocessorg(org)
            self.coauthors.append(coauthor)
            self.author_org.append(org)

    def add_edge(self):
        for i1,node1 in enumerate(self.nodes):
            for i2,node2 in enumerate(self.nodes):
                if i2<=i1:continue
                # 使用共同作者进行的判断
                len_a1, len_a2, len_a1_a2 = get_coauthor_infor(self.coauthors[i1], self.coauthors[i2])
                if (len_a1_a2 > 2) or (len_a1_a2 == 2 and max(len_a1, len_a2) <= 4):
                    self.graph.add_edge(node1, node2)
                    continue
                # 使用组织机构进行的判断
                # print(self.author_org[i1],self.author_org[i2])
                len_o1, len_o2, dist_o1_o2 = get_org_infor(self.author_org[i1], self.author_org[i2])
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

    def add_edge3(self):
        for i1, node1 in enumerate(self.nodes):
            for i2, node2 in enumerate(self.nodes):
                if i2 <= i1: continue
                # 使用共同作者进行的判断
                len_a1, len_a2, len_a1_a2 = get_coauthor_infor(self.coauthors[i1], self.coauthors[i2])
                if (len_a1_a2 > 2) or (len_a1_a2 == 2 and max(len_a1, len_a2) <= 4):
                    self.graph.add_edge(node1, node2)
                    continue
                # 使用组织机构进行的判断
                len_o1, len_o2, dist_o1_o2 = get_org_infor(self.author_org[i1], self.author_org[i2])
                if len_o1 == 0 or len_o2 == 0: continue  # 组织机构为空的不能进行判断
                if dist_o1_o2 == 0 or dist_o1_o2 < min(len_o1, len_o2) / 5:
                    self.graph.add_edge(node1, node2)
                    continue
                p_org1=process_org2(self.author_org[i1])
                p_org2=process_org2(self.author_org[i2])
                if len(set(p_org1) & set(p_org2))>=min(len(set(p_org1)),len(set(p_org1))):
                    self.graph.add_edge(node1, node2)
                # # print(self.author_org[i1], self.author_org[i2])
                # len_o1, len_o2, dist_o1_o2 = get_org_infor(self.author_org[i1].lower(), self.author_org[i2].lower())
                # if len_o1 == 0 or len_o2 == 0: continue  # 组织机构为空的不能进行判断
                # if dist_o1_o2 == 0 or dist_o1_o2 < min(len_o1, len_o2) / 2:
                #     self.graph.add_edge(node1, node2)
                #     continue

class GraphClusters(GraphBase):
    def __init__(self,nodes_info_dict):
        super().__init__(nodes_info_dict)

class GraphTitles(GraphBase):
    def __init__(self,nodes_info_dict,eps_dist=0.9):
        super().__init__(nodes_info_dict)
        self.sim_matrix=None
        self.eps_dist=eps_dist
        self.origin_cluster=None
        self.origin_dist_cluster=None

    def add_edge(self):
        if self.sim_matrix is None: return
        for i1, node1 in enumerate(self.nodes):
            for i2, node2 in enumerate(self.nodes):
                if i2 <= i1: continue
                if self.sim_matrix[i1][i2]>self.eps_dist:
                    self.graph.add_edge(node1,node2)

    def del_edge(self,dist_del=0.5):
        l1=len(self.get_connected_components())
        for i1,node1 in enumerate(self.nodes):
            for i2,node2 in enumerate(self.nodes):
                if i2<=i1:continue
                if self.graph.has_edge(node1,node2) and 1-self.sim_matrix[i1][i2]>dist_del:
                    # print('删除文章之间的联系',node1,node2,self.sim_matrix[i1][i2],dist_del)
                    self.graph.remove_edge(node1,node2)
        print('删除前的聚类数',l1,'删除后的聚类数', len(self.get_connected_components()))

    def get_dist_matrix(self,num_topic=20):
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
        lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=num_topic)
        lsi_corpus = [lsi[bc] for bc in tfidf_corpus]
        similarity_lsi = similarities.MatrixSimilarity(lsi_corpus)
        self.sim_matrix=similarity_lsi[lsi_corpus]

    def get_origin_cluster_dist(self,dist_type=1):
        init_clusters = self.get_connected_components()
        print('原始聚类数', len(init_clusters))
        dist_dict = {}
        for i1, c1 in enumerate(init_clusters):
            for i2, c2 in enumerate(init_clusters):
                if i2 <= i1: continue
                if dist_type==1:
                    dist_cluster = self.dist_two_paper_sets(c1, c2)
                elif dist_type==2:
                    dist_cluster = self.dist_two_paper_sets2(c1,c2)
                else:
                    dist_cluster = self.dist_two_paper_sets(c1, c2)
                dist_dict[(i1, i2)] = dist_cluster
        self.origin_cluster=init_clusters
        self.origin_dist_cluster=dist_dict

    def get_cluster(self,eps_dist=0.01,num_max=10):
        init_clusters=self.get_connected_components()
        if len(init_clusters)==0: return init_clusters
        
        counter=0
        while True:
            print(counter)
            counter+=1
            if counter>num_max: break
            min_dist=10000
            index1=-1
            index2=-1
            for i1,c1 in enumerate(init_clusters):
                for i2,c2 in enumerate(init_clusters):
                    if i2<=i1:continue
                    dist_cluster=self.dist_two_paper_sets(c1,c2)
                    if dist_cluster<min_dist:
                        index1=i1
                        index2=i2
                        min_dist=dist_cluster
            if min_dist<eps_dist:
                new_cluster=init_clusters[index1]+init_clusters[index2]
                init_clusters.pop(index2)
                init_clusters.pop(index1)
                init_clusters.append(new_cluster)
            else:
                break
        return init_clusters

    def get_cluster2(self,eps_dist=0.05,num_max=50):
        if self.origin_cluster is None:
            self.get_origin_cluster_dist()
        res_dist = copy.deepcopy(self.origin_dist_cluster)
        res_cluster=[[i] for i in range(len(self.origin_cluster))]
        drop_index=set()
        counter = 0
        while counter<num_max:
            counter+=1
            key,value=min(res_dist.items(), key=lambda x: x[1])
            if value>eps_dist:break # 所有的类别间距大于给定的阈值
            res_cluster[key[0]]=res_cluster[key[0]]+res_cluster[key[1]]
            # print('迭代次数',counter,'存在需要合并的类别',key[0],key[1],value)
            drop_index.add(key[1])
            for i in range(len(res_cluster)):
                for j in range(len(res_cluster)):
                    if j<=i:continue
                    if i == key[0]:
                        res_dist[(i,j)]=max(res_dist[(i,j)],res_dist[(i,key[1])])
                    if j==key[0]:
                        res_dist[(i, j)] = max(res_dist[(i, j)], res_dist[(j, key[1])])
                    if i==key[1] or j==key[1]: # key[1]已经聚到key[0] 其他到key[1]的距离设为很大
                        res_dist[(i, j)] = 1000
        # print(drop_index)
        end_cluster=[]
        for i,index_list in enumerate(res_cluster):
            if i not in drop_index:
                c=[]
                for index in index_list:
                    c+=self.origin_cluster[index]
                end_cluster.append(c)
        print('迭代次数',counter)
        return end_cluster

    def dist_two_paper_sets(self,pid1s,pid2s):
        max_dist=-100
        for i1,p1 in enumerate(pid1s):
            index1=self.nodes.index(p1)
            min_dist=10000
            for i2,p2 in enumerate(pid2s):
                index2=self.nodes.index(p2)
                if 1-self.sim_matrix[index1][index2]<min_dist:
                    min_dist=1-self.sim_matrix[index1][index2]
            if min_dist>max_dist:
                max_dist=min_dist
        return max_dist

    def dist_two_paper_sets2(self,pid1s,pid2s):
        sum_dist=0
        for i1,p1 in enumerate(pid1s):
            index1=self.nodes.index(p1)
            for i2,p2 in enumerate(pid2s):
                index2=self.nodes.index(p2)
                sum_dist+=1-self.sim_matrix[index1][index2]
        return sum_dist/(len(pid1s)*len(pid2s))
                

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
    graph_at2_dict={}
    graph_at21_dict={}
    graph_a2t21_dict={}
    
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
    # 使用作者信息1+title进行聚类，分层次进行 title计算距离针对两个集合(经过作者信息聚集过的子类)进行
    for author_select in author_list:
        print(author_select)
        graph_at2=GraphTitles(t_data[author_select],0.99)
        graph_at2.set_init_graph(graph_author_dict[author_select].graph)
        graph_at2.get_dist_matrix()
        graph_at2_dict[author_select]=graph_at2.get_cluster2(0.05,500)
        
    for author_select in author_list:
        print(author_select)
        graph_at21=GraphTitles(t_data[author_select],0.99)
        graph_at21.set_init_graph(graph_author_dict[author_select].graph)
        graph_at21.get_dist_matrix()
        graph_at21.get_origin_cluster_dist()
        graph_at21_dict[author_select]=graph_at21
    
    for author_select in author_list:
        print(author_select)
        graph_a2t21=GraphTitles(t_data[author_select],0.99)
        graph_a2t21.set_init_graph(graph_author_dict2[author_select].graph)
        graph_a2t21.get_dist_matrix()
        graph_a2t21.get_origin_cluster_dist()
        graph_a2t21_dict[author_select]=graph_a2t21

    for author_select in author_list:
        real_res={author_select:t_tag[author_select]}
        model_author={author_select: graph_author_dict[author_select].get_connected_components()}
#        model_author2={author_select: graph_author_dict2[author_select].get_connected_components()}
#        model_title={author_select:graph_title_dict[author_select].get_connected_components()}
#        model_at={author_select:graph_at_dict[author_select].get_connected_components()}
#        model_a2t={author_select:graph_a2t_dict[author_select].get_connected_components()}
#        model_at2={author_select:graph_at2_dict[author_select]}
#        model_at21={author_select:graph_at21_dict[author_select].get_cluster2(0.005,100)}
#        model_a2t21={author_select:graph_a2t21_dict[author_select].get_cluster2(0.1,500)}
        print(author_select,len(real_res[author_select]))
        print('----> model_author',
              '%.2f'%pairwise_f1(real_res,model_author),len(model_author[author_select]))
#        print('----> model_author2',
#              '%.2f'%pairwise_f1(real_res,model_author2),len(model_author2[author_select]))
#        print('----> model_title',
#              '%.2f'%pairwise_f1(real_res,model_title),len(model_title[author_select]))
#        print('----> model_at',
#              '%.2f'%pairwise_f1(real_res,model_at),len(model_at[author_select]))
#        print('----> model_a2t',
#              '%.2f'%pairwise_f1(real_res,model_a2t),len(model_a2t[author_select]))
#        print('----> model_at2',
#              '%.2f'%pairwise_f1(real_res,model_at2),len(model_at2[author_select]))
#        print('----> model_at21',
#              '%.2f'%pairwise_f1(real_res,model_at21),len(model_at21[author_select]))
#        print('----> model_a2t21',
#              '%.2f'%pairwise_f1(real_res,model_a2t21),len(model_a2t21[author_select]))




