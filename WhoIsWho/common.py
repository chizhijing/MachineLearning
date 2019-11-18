import json
import re
import itertools
import Levenshtein as lv

valid_row_data_path = 'data/sna_data/sna_valid_author_raw.json'
valid_pub_data_path = 'data/sna_data/sna_valid_pub.json'
train_author_path = 'data/train/train_author.json'
train_pub_path = 'data/train/train_pub.json'

# 数据预处理

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

def process_org2(org):
    if org=='': return org
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
    org = org.replace('Prop', 'Propagation')
    org = org.replace('Propagation.', 'Propagation')
    org = org.lower()
    org = org.split(',')
    org = [temp.strip() for temp in org if temp != '']

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

# 计算pairwise precision
def pairwise_f1(real_dict,pre_dict):
    res={}
    sum_f1=0
    for key in real_dict: # 真实分类中的每一个同名作者

        real_pair = []
        pre_pair = []
        for cluster in real_dict[key]:    # 每一个实际作者
            real_pair.extend(list(itertools.combinations(cluster,2)))
        for cluster in pre_dict[key]:  # 每一个实际作者
            pre_pair.extend(list(itertools.combinations(cluster, 2)))
        if len(pre_pair)==0 or len(real_pair)==0:
            print(key,'pre_num',len(pre_pair),'real_num',len(real_pair))
            continue
#        print(key,len(real_dict[key]),len(pre_dict[key]))
        intersection=set(real_pair) & set(pre_pair)
        precision=len(intersection)/len(pre_pair)
        recall=len(intersection)/len(real_pair)
        f1=precision*recall*2/(precision+recall)
        sum_f1+=f1
        # print(key,'precision',precision,'recall',recall,'f1',f1)
        res[key]=[intersection,precision,recall,f1]
    if len(res)==0: return -1
#    print(sum_f1 / len(res))
    return sum_f1/len(res)
# 计算pairwise precision
def pairwise_f1_new(real_dict,pre_dict):
    res={}
    sum_f1=0
    for key in real_dict: # 真实分类中的每一个同名作者
        real_pair = []
        pre_pair = []
        for cluster in real_dict[key]:    # 每一个实际作者
            self_pair=[(p,p) for p in cluster]
            real_pair.extend(list(itertools.combinations(cluster,2)))
            real_pair.extend(self_pair)
        for cluster in pre_dict[key]:  # 每一个实际作者
            self_pair=[(p,p) for p in cluster]
            pre_pair.extend(list(itertools.combinations(cluster, 2)))
            pre_pair.extend(self_pair)
        if len(pre_pair)==0 or len(real_pair)==0:
            print(key,'pre_num',len(pre_pair),'real_num',len(real_pair))
            continue
#        print(key,len(real_dict[key]),len(pre_dict[key]))
        intersection=set(real_pair) & set(pre_pair)
        precision=len(intersection)/len(pre_pair)
        recall=len(intersection)/len(real_pair)
        f1=precision*recall*2/(precision+recall)
        sum_f1+=f1
        # print(key,'precision',precision,'recall',recall,'f1',f1)
        res[key]=[intersection,precision,recall,f1]
    if len(res)==0: return -1
#    print(sum_f1 / len(res))
    return sum_f1/len(res)

# 计算pairwise recall
def pairwise_r(real,predict):
    pass

# 获取训练集数据
def get_train_data(author_list=None):
    """
    train_tag:{author_name:[[papers_cluster1],[papers_cluster2]...],...}
    train_data:{author_name:[{paper_detail1},{paper_detail2},...],...}
    """
    train_author_data = json.load(open(train_author_path, 'r', encoding='utf-8'))
    train_pub_data = json.load(open(train_pub_path, 'r', encoding='utf-8'))
    train_data = {}
    train_tag={}
    if author_list is None:
        author_list=list(train_author_data.keys())
    for author in author_list:
        pid_list = [train_pub_data[p] for pp in train_author_data[author].values() for p in pp]
        train_data[author] = pid_list
        train_tag[author]=train_author_data[author].values()
#    train_tag = {key: [pid for pid in train_author_data[key].values()] for key in list(train_author_data.keys())}  # 训练数据真实标签
    return train_data,train_tag

def get_valid_data():
    validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
    validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
    for author in validate_data:
        validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]
    return validate_data

# 返回给定共同作者列表中，需要消歧作者的组织机构
def get_author_org(co_list,author_name):
    for co_author in co_list:
        if precessname(co_author['name'])==author_name:
            if 'org' in co_author.keys():
                return co_author['org']
            else:
                return ''
    return ''

def get_coauthor_infor(co_list1,co_list2):
    name1_list=[precessname(c1['name']) for c1 in co_list1]
    name2_list=[precessname(c2['name']) for c2 in co_list2]
    return len(name1_list),len(name2_list),len(set(name1_list) & set(name2_list))

def get_org_infor(org1,org2):
    p_org1=preprocessorg(org1)
    p_org2=preprocessorg(org2)
    return len(p_org1),len(p_org2),lv.distance(p_org1,p_org2)

if __name__ == '__main__':
    r_dict={'a':[[1,2,3],[5,6]],'b':[[3,4],[7]],'c':[[8,9,10]]}
    p_dict = {'a': [[1, 2, 5], [3], [6]], 'b': [[3, 4], [7]], 'c': [[8, 9], [10]]}
    p_dict2={'a':[[1,2],[3],[5],[6]],'b':[[3,4],[7]],'c':[[8,9],[10]]}
    pairwise_f1(r_dict,p_dict)
    pairwise_f1(r_dict, p_dict2)

    # train_author_data = json.load(open(train_author_path, 'r', encoding='utf-8'))
    # for author_name in train_author_data.keys():
    #     print(author_name,len(train_author_data[author_name]))



