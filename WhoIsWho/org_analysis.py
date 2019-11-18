from graph import get_train_data,precessname
from common import process_org2
from collections import defaultdict

if __name__=='__main__':
    # 取训练集数据
    train_data,train_tag=get_train_data()

    # 获取org信息
    author_org={}
    for author,paper_ids in train_tag.items(): # 遍历同名作者
        if author != 'li_guo':continue
        author_org_list = []
        for paper_id_list in paper_ids: # 遍历实际作者
#            print(paper_id_list)
            org_list = []
            for pid in paper_id_list: # 遍历文章
#                print(author,pid)
                paper=train_data[author][pid]
                org = ''
                if 'authors' in paper.keys():
                    for co_author in paper['authors']:
                        if precessname(co_author['name']) == author:
                            org=co_author['org']
                            org_p=process_org2(org)
                            break
                if len(org) != 0: org_list.extend(org_p)
            if len(org_list)!=0: author_org_list.append(org_list)
        author_org[author]=author_org_list
        
        # 计算词频
        freq = defaultdict(int)
        for orgs in author_org_list:
            for org in orgs:
                freq[org] += 1
       
        for w in freq.keys():
            if freq[w]>=2:
                print(w,freq[w])