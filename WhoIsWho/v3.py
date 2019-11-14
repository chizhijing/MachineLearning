from common import *
import json
import networkx as nx

# 计算两篇文章共同作者数
def num_coauthor_paper(paper1,paper2):
#    author1s = [paper_author['name'] for paper_author in paper1['authors']]
#    author2s = [paper_author['name'] for paper_author in paper2['authors']]
    author1s = [precessname(paper_author['name']) for paper_author in paper1['authors']]
    author2s = [precessname(paper_author['name']) for paper_author in paper2['authors']]
    num_coauthor=len(set(author1s) & set(author2s))
    return num_coauthor

# 计算两篇文章的工作组织机构数
def num_org_paper(paper1,paper2):
    org1s=[preprocessorg(paper_author['org']) for paper_author in paper1['authors']]
    org2s=[preprocessorg(paper_author['org']) for paper_author in paper1['authors']]
    num_org=len(set(org1s)&set(org2s))
    return num_org
    
    # 计算联通子图(具有共同作者的文章集合)
def graph_operate(papers):
    graph=nx.Graph()
    pids=[paper['id'] for paper in papers]
    graph.add_nodes_from(pids)
    for index1,p1 in enumerate(papers):
        if index1==len(papers)-1:break
        for index2,p2 in enumerate(papers[index1+1:]):
            num_co_au=num_coauthor_paper(p1,p2)
            if num_co_au>=2:
                graph.add_edge(p1['id'],p2['id'])
#            else:
#                if num_org_paper(p1,p2)>=1:
#                    graph.add_edge(p1['id'],p2['id'])
    conn_comp=list(nx.connected_components(graph))
    conn_comp=[list(c) for c in conn_comp]
#    print(conn_comp)
    return conn_comp

def disambiguate_by_graph(validate_data):
    res_dict={}
    print('不同名作者数',len(validate_data))
    for i,author in enumerate(validate_data.keys()):
        author_papers=validate_data[author]
        print(i,author,len(author_papers))
        author_cluster=graph_operate(author_papers)
        print(i,author,'文章数',len(author_papers),'消歧后作者数',len(author_cluster))
        res_dict[author]=author_cluster
    return res_dict


if __name__ == '__main__':
    pass
#     validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
#     validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
#     merge_data = {}
#     for author in validate_data:
#         validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]
#    
#     res=disambiguate_by_graph(validate_data)
#     json.dump(res, open('result/disambiguate_by_graph4.json', 'w', encoding='utf-8'), indent=4)
#     
#
#    train_author_data = json.load(open(train_author_path, 'r', encoding='utf-8'))
#    train_pub_data = json.load(open(train_pub_path, 'r', encoding='utf-8'))
#    merge_data={}
#    for author_name in train_author_data.keys():
#        p_list=[]
#        for a_id in train_author_data[author_name].keys():
#            p_list.extend(train_author_data[author_name][a_id])
#        merge_data[author_name]=[train_pub_data[p_id] for p_id in p_list]
#    
#    res_cluster=disambiguate_by_graph(merge_data)
#    res_real={key:list(train_author_data[key].values()) for key in train_author_data.keys()}
#
#    liguo_cluster=disambiguate_by_graph()
#    
##    author_select='li_guo'
##    m_data={author_select:merge_data[author_select]}
#    res_real={author_select:list(train_author_data[author_select].values())}
#    print('真实作者数',len(res_real[author_select]))
#    res_cluster=disambiguate_by_graph(merge_data)
#    pairwise_f1(res_real,res_cluster)
