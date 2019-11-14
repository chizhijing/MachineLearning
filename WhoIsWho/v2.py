import json
import re
import numpy as np
import networkx as nx
import random

import matplotlib.pyplot as plt
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

def disambiguate_by_xxx(validate_data):
    res_dict = {}

    for author in validate_data:
        print(author)
        res = []
        papers = validate_data[author]
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            d = {}
            authors = [precessname(paper_author['name']) for paper_author in paper['authors']]
            if author in authors:
                authors.remove(author)
            org = preprocessorg(get_org(paper['authors'], author))
            venue = paper['venue']
            d["authors"] = authors
            d["org"] = org
            d['keywords'] = paper['keywords'] if 'keywords' in paper else ""
            d['venue'] = venue

            if len(res) == 0:
                res.append([paper['id']])
            else:
                max_inter = 0
                indx = 0
                for i, clusters in enumerate(res):
                    score = 0
                    for pid in clusters:
                        insection = set(paper_dict[pid]['authors']) & set(authors)
                        score += len(insection)

                    # if org != "" and (org in paper_dict[pid]['org'] or paper_dict[pid]['org'] in org):
                    #                             score += 10

                    if score > max_inter:
                        max_inter = score
                        indx = i

                if max_inter > 0:
                    res[indx].append(paper['id'])  # 若最高分大于0，将id添加到得分最高的簇中
                else:
                    res.append([paper['id']])  # 否则，另起一簇

            paper_dict[paper['id']] = d

        res_dict[author] = res
        
        return res_dict

# 计算两篇文章之间的距离
def dist_paper(paper1,paper2):
#    print('in dist paper')
#    print(paper1['authors'],paper2['authors'])
    author1s = [paper_author['name'] for paper_author in paper1['authors']]
    author2s = [paper_author['name'] for paper_author in paper2['authors']]
    num_coauthor=len(set(author1s) & set(author2s))
    if num_coauthor>=2: return 0
    else: return 1000

# 计算两个文章列表之间的距离(使用hausdorff距离)
def dist_papers(papers1,papers2):
#    print('in dist papers')
    min_d_list = []
    for p1 in papers1:
        min_d = 10000
        for p2 in papers2:
            new_dist=dist_paper(p1,p2)
            if new_dist<min_d:
                min_d=new_dist
        min_d_list.append(min_d)
    return max(min_d_list)

# 计算两篇文章共同作者数
def num_coauthor_paper(paper1,paper2):
    author1s = [paper_author['name'] for paper_author in paper1['authors']]
    author2s = [paper_author['name'] for paper_author in paper2['authors']]
    num_coauthor=len(set(author1s) & set(author2s))
    return num_coauthor

def num_coauthor_matrix(papers):
    l=np.zeros([len(papers),len(papers)],dtype=int)
    for i in range(len(papers)-1):
        print(i)
        for j in range(i+1,len(papers)):
            l[i][j]=num_coauthor_paper(papers[i],papers[j])
    return l

def disambiguate_cluster(papers):
    print("进行同名消歧聚类")
    # 初始化每一篇文章为一个簇
    res=[[papers[pid]] for pid in papers]
    print('初始簇数目',len(res),res[0],res[1])
    # 循环合并相近的簇
    counter=0
    while counter<1000:
        counter+=1
        min_dist=10000 # 记录最小距离
        min_index1=0 # 最小距离的第1个簇index
        min_index2 = 0  # 最小距离的第2个簇index
        print("迭代次数",counter,'簇数',len(res))
        for i1, c1 in enumerate(res):
            if i1==len(res)-1:break
            for i2,c2 in enumerate(res[i1+1:]):
                new_dist=dist_papers(c1,c2)
                if new_dist<min_dist:
                    min_dist=new_dist
                    min_index1=i1
                    min_index2=i1+1+i2
                    print('更新最小距离',min_index1,min_index2,min_dist)
                if min_dist==0:break
            if min_dist==0: break
        if min_dist>1:break
        else:
            res[min_index1].extend(res[min_index2])
            res.pop(min_index2)
    return res

def graph_operate(papers):
    graph=nx.Graph()
    pids=[paper['id'] for paper in papers]
    graph.add_nodes_from(pids)
    for index1,p1 in enumerate(papers):
        if index1==len(papers)-1:break
        for index2,p2 in enumerate(papers[index1+1:]):
            num_co_au=num_coauthor_paper(p1,p2)
            if num_co_au>=3:graph.add_edge(p1['id'],p2['id'])
    conn_comp=list(nx.connected_components(graph))
    conn_comp=[list(c) for c in conn_comp]
#    print(conn_comp)
    return conn_comp

def disambiguate_by_graph(validate_data):
    res_dict={}
    for author in validate_data.keys():
        author_papers=validate_data[author]
        author_cluster=graph_operate(author_papers)
        print(author,'文章数',len(author_papers),'消歧后作者数',len(author))
        res_dict[author]=author_cluster
    json.dump(res_dict, open('result/disambiguate_by_graph3.json', 'w', encoding='utf-8'), indent=4)

if __name__=="__main__":
    train_author_path = 'data/train/train_author.json'
    train_pub_path = 'data/train/train_pub.json'
    train_author_data=json.load(open(train_author_path,'r',encoding='utf-8'))
    train_pub_data=json.load(open(train_pub_path,'r',encoding='utf-8'))

    author_liguo=train_author_data['li_guo']
    paper_liguo={p_id:train_pub_data[p_id] for a_id in author_liguo.keys() for p_id in author_liguo[a_id]}

    ress=disambiguate_by_xxx(paper_liguo)
    result=disambiguate_cluster(paper_liguo)

    rand_pid = random.sample(paper_liguo.keys(), 100)
    rand_paper = {key: paper_liguo[key] for key in rand_pid}
    xxx = graph_operate(paper_liguo)

# 验证集数据
    valid_row_data_path = 'data/sna_data/sna_valid_author_raw.json'
    valid_pub_data_path = 'data/sna_data/sna_valid_pub.json'

    # 合并数据
    validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
    validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
    merge_data = {}
    for author in validate_data:
        validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]
    disambiguate_by_graph(validate_data)
    
    
