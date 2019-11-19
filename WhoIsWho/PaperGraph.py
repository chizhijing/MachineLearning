import networkx as nx
from networkx import Graph
import json
import re

sample_data_path = 'data/sample_train_data.json'
sample_tag_path = 'data/sample_train_tag.json'

# 获取sample数据
def get_sample_data():
    sample_data = json.load(open(sample_data_path, 'r', encoding='utf-8'))
    sample_tag = json.load(open(sample_tag_path, 'r', encoding='utf-8'))
    return sample_data,sample_tag

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
        org=org.lower()
        org=org.replace('electron.','electronic')
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
            self.add_node(pid,title=p_title,authors=p_authors)

    def node_at(self,index,detail=None):
        pid=self.get_pid_at(index)
        if detail:
            return self.node[pid][detail]
        return self.node[pid]

    def get_pid_at(self,index):
        node_list = list(self.nodes)
        return node_list[index]

    def dist1(self,p1,p2):
        author1=self.node[p1]['authors']
        author2=self.node[p2]['authors']
        co_num=len(set(author1.keys())&set(author2.keys()))
        # print(author1.keys(),author2.keys(),set(author1.keys()),co_num)
        return co_num


if __name__=='__main__':
    data,tag=get_sample_data()
    p_graph=PaperGraph(name='li_guo')
    p_graph.set_paper_info(data['li_guo'])