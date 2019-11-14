# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:24:40 2019

@author: chizj
"""
import networkx as nx
from common import get_train_data,pairwise_f1
import copy
from analysis02 import get_author_org,get_org_infor,get_coauthor_infor


class PaperGraph:
    def __init__(self,author_name,paper_list,tag=None):
        self.author_name=author_name
        self.tag=tag
        self.titles=[]
        self.abstracts=[]
        self.keywords=[]
        self.paper_ids=[]
        self.authors=[]
        self.venues=[]
        self.years=[]
        self.get_paper_detail(paper_list)
        self.origin_graph = self.init_graph()
        self.nlp_titles=[]

    def get_paper_detail(self,paper_list):
        for paper in paper_list:
            self.titles.append(paper['title'])
            self.paper_ids.append(paper['id'])
            self.abstracts.append(paper['abstract'] if 'abstract' in paper.keys() and paper['abstract'] is not None else '')
            self.keywords.append(
                paper['keywords'] if 'keywords' in paper.keys() and paper['keywords'] is not None else '')
            self.authors.append(
                paper['authors'] if 'authors' in paper.keys() and paper['authors'] is not None else '')
            self.venues.append(
                paper['venue'] if 'venue' in paper.keys() and paper['venue'] is not None else '')
            self.years.append(
                paper['year'] if 'year' in paper.keys() and paper['year'] is not None else '')

    def init_graph(self):
        graph=nx.Graph()
        graph.add_nodes_from(self.paper_ids)
        return graph

    def get_graph_by_coauthor(self,graph):
        """
        在已有graph的基础上，根据authors信息进行构建更新的图模型
        :return:
        """
        new_graph=copy.deepcopy(graph)
        # 具有共同作者的进行聚类
        for index1, coauthor1 in enumerate(self.authors):
            if index1 == len(self.authors) - 1: break
            org1 = get_author_org(coauthor1, self.author_name)
            for index2, coauthor2 in enumerate(self.authors):
                if index2 <= index1: continue
                # 使用共同作者进行的判断
                len_a1, len_a2, len_a1_a2 = get_coauthor_infor(coauthor1, coauthor2)
                if len_a1_a2 > 2:
                    new_graph.add_edge(self.paper_ids[index1], self.paper_ids[index2])
                    continue
                else:
                    if len_a1_a2 == 2 and max(len_a1, len_a2) <= 4:
                        new_graph.add_edge(self.paper_ids[index1], self.paper_ids[index2])
                        continue
                # 使用组织机构进行的判断
                org2 = get_author_org(coauthor2, self.author_name)
                len_o1, len_o2, dist_o1_o2 = get_org_infor(org1, org2)
                if len_o1 == 0 or len_o2 == 0: continue  # 组织机构为空的不能进行判断
                if dist_o1_o2 == 0:
                    new_graph.add_edge(self.paper_ids[index1], self.paper_ids[index2])
                    continue
                if dist_o1_o2 < min(len_o1, len_o2) / 5:
                    new_graph.add_edge(self.paper_ids[index1], self.paper_ids[index2])
                    continue
        return new_graph

    def get_graph_by_title(self,graph):
        pass

    @staticmethod
    def get_graph_components(graph):
        """
        返回联通子图(聚类结果)
        :param graph:
        :return:
        """
        conn_comp = list(nx.connected_components(graph))
        conn_comp = [list(c) for c in conn_comp]
        return conn_comp

    def get_score(self,graph):
        """
        计算给定图的得分
        :param graph:
        :return:
        """
        real_cluster={self.author_name: self.tag}
        model_cluster={self.author_name: self.get_graph_components(graph)}
        f1 = pairwise_f1(real_cluster, model_cluster)
        print(self.author_name, 'RealNum', len(real_cluster[self.author_name]),
              'PreNum', len(model_cluster[self.author_name]),'F1','%.2f' % f1)
        return len(real_cluster[self.author_name]),len(model_cluster[self.author_name]),f1

    def nlp_prepare(self):
        pass

    def get_titles(self):
        return self.titles

if __name__=='__main__':
    author_list=['li_guo']
    train_data,train_tag=get_train_data(author_list)

    author_select=author_list[0]
    
    p_graph=PaperGraph(author_select,train_data[author_select],train_tag[author_select])
