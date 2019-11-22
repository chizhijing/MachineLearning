import networkx as nx
from networkx import Graph
import json
import re
import copy
from common import valid_pub_data_path,valid_row_data_path
import Levenshtein as lv

sample_data_path = 'data/sample_train_data.json'
sample_tag_path = 'data/sample_train_tag.json'

# 获取sample数据
def get_sample_data():
    sample_data = json.load(open(sample_data_path, 'r', encoding='utf-8'))
    sample_tag = json.load(open(sample_tag_path, 'r', encoding='utf-8'))
    return sample_data,sample_tag

def get_valid_data():
    validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
    validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
    for author in validate_data:
        validate_data[author] = {paper_id:validate_pub_data[paper_id] for paper_id in validate_data[author]}
    return validate_data

def process_title(title):
    return title.lower()

def process_authors(authors):
    author_dict={}
    for author_info in authors:
        # 姓名预处理
        name=author_info['name']
        name = name.lower().replace(' ', '_')
        name = name.replace('.', '_')
        name = name.replace('-', '')
        name = re.sub(r"_{2,}", "_", name)

        # 组织机构预处理
        if 'org' in author_info.keys() and author_info['org']:
            org=author_info['org']
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
            org = org.replace('C.', 'Center')
            org = org.replace('Ophthal.', 'Ophthalmic')
            org = org.replace('Prop.', 'Propagation')
            org = org.replace(' & ',' and ')
            # org = org.replace(', ', ',')
            # org = org.replace(' ,', ',')
            org=org.lower()
            org=org.replace('electron.','electronic')
            org=re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", org)
        else:
            org=''
        # 保存为字典
        author_dict[name]=org
    return author_dict


class PaperGraph(Graph):
    def set_paper_info(self,paper_dict):
        for pid,paper_info in paper_dict.items():
            p_title=process_title(paper_info['title'])
            if 'authors' in paper_info.keys():
                p_authors=process_authors(paper_info['authors'])
            else:
                p_authors={}
            author_org=''
            if len(p_authors)>0:
                if self.name in p_authors.keys():
                    author_org=p_authors[self.name]
                else:
                    p_authors_new={}
                    for author,org in p_authors.items():
                        s_name = author.split('_')
                        s_name.sort()
                        s_name='_'.join(s_name)
                        p_authors_new[s_name]=org
                    author_name=self.name.split('_')
                    author_name.sort()
                    author_name='_'.join(author_name)
                    if author_name in p_authors_new.keys():
                        # print(self.name,p_authors)
                        # print(author_name,p_authors_new)
                        author_org = p_authors_new[author_name]
            self.add_node(pid,title=p_title,authors=p_authors,author_org=author_org)

    def cal_node_pair_info(self):
        self.num_coauthor_dict={}
        self.sim_author_org_dict={}
        self.sim_title_dict={}
        for i1, node1 in enumerate(self.nodes):
            for i2, node2 in enumerate(self.nodes):
                if i2 <= i1: continue
                author1 = self.node[node1]['authors']
                author2 = self.node[node2]['authors']
                name1_set = set(author1.keys())
                name2_set = set(author2.keys())
                name12_set = name1_set & name2_set
                self.num_coauthor_dict[(node1,node2)]=(len(name1_set),len(name2_set),len(name12_set))

                org1 = self.node[node1]['author_org']
                org2 = self.node[node2]['author_org']
                sim_org=self.pair_jaccard(org1,org2)
                self.sim_author_org_dict[(node1,node2)]=sim_org

                self.sim_title_dict[(node1,node2)]=self.pair_jaccard(self.node[node1]['title'],self.node[node2]['title'])

    def node_at(self,index,detail=None):
        pid=self.get_pid_at(index)
        if detail:
            return self.node[pid][detail]
        return self.node[pid]

    def get_pid_at(self,index):
        node_list = list(self.nodes)
        return node_list[index]

    def get_res1(self):
        graph1=nx.Graph()
        graph1.add_nodes_from(self.nodes)
        for i1,node1 in enumerate(graph1.nodes):
            for i2,node2 in enumerate(graph1.nodes):
                if i2<=i1:continue
                # 使用共同作者进行的判断
                author1 = self.node[node1]['authors']
                author2 = self.node[node2]['authors']
                name1_set = set(author1.keys())
                name2_set = set(author2.keys())
                name12_set=name1_set&name2_set

                if (len(name12_set) > 2) or (len(name12_set) == 2 and max(len(name1_set), len(name2_set)) <= 4):
                    graph1.add_edge(node1, node2)
                    continue
                # 使用组织机构进行的判断
                org1 = self.node[node1]['author_org']
                org2 = self.node[node2]['author_org']
                if self.is_same_org(org1, org2):
                    # print(org1)
                    graph1.add_edge(node1, node2)
                    continue

        for i1,node1 in enumerate(graph1.nodes):
            for i2,node2 in enumerate(graph1.nodes):
                if i2<=i1 or not graph1.has_edge(node1, node2):continue
                # 删除操作
                org1 = self.node[node1]['author_org']
                org2 = self.node[node2]['author_org']
                if  self.org_dist(org1, org2) > 0.95:
                    graph1.remove_edge(node1, node2)

        conn_comp = list(nx.connected_components(graph1))
        conn_comp = [list(c) for c in conn_comp]
        return conn_comp

    def get_res2(self):
        graph1 = nx.Graph()
        graph1.add_nodes_from(self.nodes)
        for i1, node1 in enumerate(graph1.nodes):
            for i2, node2 in enumerate(graph1.nodes):
                if i2 <= i1: continue
                # 添加联系
                l1,l2,l12=self.num_coauthor_dict[(node1,node2)]
                sim = self.sim_author_org_dict[(node1, node2)]
                if (l12 > 2) or (l12 == 2 and max(l1, l2) <= 4): # 使用共同作者进行的判断
                    graph1.add_edge(node1, node2)
                elif sim> 0.9:  # 使用组织机构进行的判断
                    graph1.add_edge(node1,node2)

                # 删除联系
                if graph1.has_edge(node1,node2):
                    org1 = self.node[node1]['author_org']
                    org2 = self.node[node2]['author_org']
                    sim_title = self.sim_title_dict[(node1, node2)]
                    if sim < 0.05 and org1!='' and org2!='':
                        graph1.remove_edge(node1, node2)
                    elif sim_title<0.05:
                        graph1.remove_edge(node1,node2)

        conn_comp = list(nx.connected_components(graph1))
        conn_comp = [list(c) for c in conn_comp]
        return conn_comp

    def get_res3(self):
        graph1 = nx.Graph()
        graph1.add_nodes_from(self.nodes)
        for i1, node1 in enumerate(graph1.nodes):
            for i2, node2 in enumerate(graph1.nodes):
                if i2 <= i1: continue
                # 添加联系
                if self.paper_is_same_author(node1,node2):
                    graph1.add_edge(node1,node2)
                # 删除联系

        conn_comp = list(nx.connected_components(graph1))
        conn_comp = [list(c) for c in conn_comp]
        return conn_comp

    def paper_is_same_author(self,pid1,pid2):
        l1, l2, l12 = self.num_coauthor_dict[(pid1, pid2)]
        sim_author_org = self.sim_author_org_dict[(pid1, pid2)]
        sim_title=self.sim_title_dict[(pid1,pid2)]
        if sim_author_org == 1:return True
        if l12 >= 3: return True
        if l12 ==2:
            if sim_author_org>0.5 and sim_author_org>0.3: return True
        # if sim_title>0.5:return True
        return False

    def is_same_org(self,org1,org2):
        if org1=='' or org2=='':return False
        if org1==org2: return True
        set1 = set(re.split(r'[;,\s]\s*', org1))
        set2 = set(re.split(r'[;,\s]\s*', org2))
        if len(set1&set2)/len(set1|set2)>0.9:
            # print(org1)
            # print(org2)
            return True
        # if len(set1&set2)==min(len(set1),len(set2)) and len(set1&set2)>=15:
        #     # print('*****')
        #     # print(org1)
        #     # print(org2)
        #     # print('####')
        #     return True

    def org_dist(self,org1,org2):
        if org1=='' or org2=='':return 0
        if org1==org2: return 0
        set1 = set(re.split(r'[;,\s]\s*', org1))
        set2 = set(re.split(r'[;,\s]\s*', org2))
        return 1-len(set1&set2)/len(set1|set2)

    def pair_jaccard(self,str1,str2):
        if str1=='' or str2=='':return -1
        if str1==str2: return 1
        set1 = set(re.split(r'[;,\s]\s*', str1))
        set2 = set(re.split(r'[;,\s]\s*', str2))
        return len(set1&set2)/len(set1|set2)

    def dist1(self,p1,p2):
        author1=self.node[p1]['authors']
        author2=self.node[p2]['authors']
        name1_set=set(author1.keys())
        name2_set = set(author2.keys())
        name1_set_new=set()
        name2_set_new = set()
        for name1 in name1_set:
            s_name=name1.split('_')
            s_name.sort()
            name1_set_new.add('_'.join(s_name))
        for name2 in name2_set:
            s_name = name2.split('_')
            s_name.sort()
            name2_set_new.add('_'.join(s_name))

        co_num1=len(name1_set&name2_set)
        co_num2=len(name1_set_new&name2_set_new)
        if co_num1<2<=co_num2:
            print(co_num1, name1_set, name2_set)
            print(co_num2, name1_set_new, name2_set_new)
            print(name1_set_new&name2_set_new)
        # if self.name not in name1_set or self.name not in name2_set:
        #     print(co_num,name1_set,name2_set)
        # print(author1.keys(),author2.keys(),set(author1.keys()),co_num)
        return co_num1,co_num2


if __name__=='__main__':
    data,tag=get_sample_data()
    p_graph=PaperGraph(name='li_guo')
    p_graph.set_paper_info(data['li_guo'])