import re
import json
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


# 1. 基于规则（按组织机构消歧） 线上得分：0.1165
def disambiguate_by_org(validate_data):
    res_dict = {}
    # 遍历同名作者
    for author in validate_data:
        res_dict[author] = []

        print(author)
        papers = validate_data[author]
        org_dict = {}
        org_dict_coauthor = {}
        no_org_list = []
        # 遍历同名作者的文章
        for paper in papers:
            authors = paper['authors']
            # 遍历某篇文章的所有作者
            for paper_author in authors:
                name = precessname(paper_author['name'])
                org = preprocessorg(paper_author['org']) if 'org' in paper_author else ""
                name = name.split('_')

                if '_'.join(name) == author or '_'.join(name[::-1]) == author:
                    if org == "":
                        no_org_list.append(
                            (paper['id'], [precessname(paper_author['name']) for paper_author in authors]))
                    else:  # 按组织聚类
                        if org not in org_dict:
                            org_dict[org] = [paper['id']]
                            org_dict_coauthor[org] = [precessname(paper_author['name']) for paper_author in authors]
                        else:
                            org_dict[org].append(paper['id'])
                            org_dict_coauthor[org].extend(
                                [precessname(paper_author['name']) for paper_author in authors])

        # 没有组织的根据合作者交集
        for p_id, names in no_org_list:
            tmp = ""
            max_num = 1
            for org in org_dict_coauthor:
                set_a = set(names)
                set_b = set(org_dict_coauthor[org])
                intersection = set_a & set_b
                iou = len(intersection)
                if iou > max_num:
                    max_num = iou
                    tmp = org

            if max_num != 1:
                org_dict[tmp].append(p_id)
            else:
                res_dict[author].append([p_id])

        for org in org_dict:
            res_dict[author].append(org_dict[org])
        json.dump(res_dict, open('result/disambiguate_by_org_result.json', 'w', encoding='utf-8'), indent=4)


# 2. 基于规则（按合作者与组织机构消歧） 线上得分：0.2449
def disambiguate_by_coauthor(validate_data):
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
    json.dump(res_dict, open('result/disambiguate_by_coauthor_result.json', 'w', encoding='utf-8'), indent=4)


def disambiguate_test(validate_data):
    for author in validate_data: # 遍历所有的同名作者
        if author!='heng_li': continue
        print('同名作者:',author,'文章数:',len(validate_data[author]))
        papers=validate_data[author]
        res=[] # 分类的结果
        for paper in papers:    # 遍历同名作者的所有文章
            d={}
            if len(res)==0:
                res.append([paper['id']])
            else:
                score=0 # 得分
                max_score=0 # 最高得分
                index=0 # 类别索引
                for i, cluster in enumerate(res):   # 遍历已有的簇
                    for pid in cluster: # 遍历簇中的每一篇文章
                        pass


def has_coauthor(paper1,paper2):
    authors1=paper1['author']
    authors2=paper2['author']

    for a1 in authors1:
        for a2 in authors2:
            if a1==a2:return True;
    return False;



if __name__=="__main__":
    valid_row_data_path = 'data/sna_data/sna_valid_author_raw.json'
    valid_pub_data_path = 'data/sna_data/sna_valid_pub.json'

    # 合并数据
    validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
    validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
    merge_data = {}
    for author in validate_data:
        validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]

    # disambiguate_by_org(validate_data)

    disambiguate_by_coauthor(validate_data)