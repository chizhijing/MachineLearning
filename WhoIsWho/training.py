from PaperGraph import get_sample_data
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
from common import pairwise_f1

def generate_character_data(author_name):
    file_path='./data/character_'+author_name+'.csv'
    if os.path.exists(file_path):
        print('载入计算好的特征')
        df=pd.read_csv(file_path,index_col=[0,1])    
        return df
    data,tag=get_sample_data()
    res_dict = {'x1':{},'x2':{},'x3':{},'x4':{},'x5':{},'label':{}}
    author_paper_pair={}
    print('开始计算标签...')
    # 同名作者的所有文章之间先赋值为0(不属于同一个实际作者)
    for author,cluster in tag.items():
        if author!=author_name:continue
        print(author)
        cluster_flatten=sum(cluster,[])
        for i1,p1 in enumerate(cluster_flatten):
            for i2,p2 in enumerate(cluster_flatten):
                if i2<=i1:continue
                res_dict['label'][(p1,p2)]=0
                author_paper_pair[(p1,p2)]=author

    # 同一个实际作者的文章之间赋值为1
    for author,cluster in tag.items():
        if author != author_name: continue
        print(author)
        for cluster_i in cluster:
            for i1,p1 in enumerate(cluster_i):
                for i2,p2 in enumerate(cluster_i):
                    if i2<=i1:continue
                    res_dict['label'][(p1, p2)]=1
    print('结束计算标签')
    print('开始计算特征...')
    # 计算character
    all_paper_dict={}
    for paper_dict in data.values():
        all_paper_dict.update(paper_dict)

    for (p1,p2) in res_dict['label'].keys():
        l1,l2,l12,sl12,sim_org=cal_paper_pair_character(all_paper_dict[p1],all_paper_dict[p2],author_paper_pair[(p1,p2)])
        res_dict['x1'][(p1,p2)]=l1
        res_dict['x2'][(p1,p2)] = l2
        res_dict['x3'] [(p1,p2)]= l12
        res_dict['x4'][(p1, p2)] = sl12
        res_dict['x5'][(p1, p2)] = sim_org
    df = pd.DataFrame(res_dict)
    df.to_csv(file_path)
    return df

def cal_paper_pair_character(p1,p2,author):
    """
    计算两篇文章之间的特征
    :param p1:
    :param p2:
    :param author:
    :return:
    """
    # 计算共同作者数
    authors1=pre_process_authors(p1)
    authors2=pre_process_authors(p2)
    name_set1=set(authors1.keys())
    name_set2=set(authors2.keys())
    # 姓名按字母排序
    authors1_new = sort_author_name(authors1)
    authors2_new = sort_author_name(authors2)
    sort_name_set1 = set(authors1_new.keys())
    sort_name_set2 = set(authors2_new.keys())
    # 当前作者组织机构的相似度
    sim_author_org=0
    author_name_sort = author.split('_')
    author_name_sort.sort()
    author_name_sort = '_'.join(author_name_sort)
    if author_name_sort in sort_name_set1&sort_name_set2:
        sim_author_org=pair_jaccard_sim(authors1_new[author_name_sort],authors2_new[author_name_sort])

    return len(name_set1),len(name_set2),len(name_set1&name_set2),len(sort_name_set1&sort_name_set2),sim_author_org

def sort_author_name(author_dict):
    authors_new={}
    for author,org in author_dict.items():
        s_name = author.split('_')
        s_name.sort()
        s_name = '_'.join(s_name)
        authors_new[s_name] = org
    return authors_new

def pair_jaccard_sim(str1,str2):
    if str1=='' or str2=='':return 0
    if str1==str2: return 10
    set1 = set(re.split(r'[;,\s]\s*', str1))
    set2 = set(re.split(r'[;,\s]\s*', str2))
    return int(len(set1&set2)/len(set1|set2)*10)

def pre_process_authors(paper_info):
    if 'authors' not in paper_info.keys():
        return None
    author_dict={}
    for author_info in paper_info['authors']:
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


if __name__ == '__main__':
    data,tag=get_sample_data()
    author_list=['li_guo', 'bo_shen', 'di_wang', 'long_wang', 'qiang_xu', 
                 'xiang_wang', 'changming_liu', 'kenji_kaneko', 'guohua_chen', 'hai_jin', 
                 'jia_li', 'guoliang_li', 'lan_wang', 'alessandro_giuliani', 'jiang_he', 
                 'xiang_gao', 'jianping_wu', 'peng_shi', 'feng_wu', 'jing_zhu']
    author='li_guo'
    df=generate_character_data(author)

    x = df[['x1', 'x2', 'x3', 'x4', 'x5']].values
    y = df['label'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)

    from sklearn_models import naive_bayes_classifier
    from sklearn_models import knn_classifier
    from sklearn_models import logistic_regression_classifier
    from sklearn_models import random_forest_classifier
    from sklearn_models import decision_tree_classifier
    from sklearn_models import svm_classifier
    from sklearn_models import svm_cross_validation
    from sklearn_models import gradient_boosting_classifier

    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }
    
    for key,classifier in classifiers.items():
        if key not in ['RF']:continue
        print('Model-',key)
        model= classifier(x_train,y_train)
        y_pre=model.predict(x_test)
        y_pre_prob=model.predict_proba(x_test)
        cr=metrics.classification_report(y_test,y_pre,labels=[0,1],target_names=['diff','same'])
        print(cr)
    
    # 构造相似度矩阵
    p1=set([df.index[i][0] for i in range(len(df))])
    p2=set([df.index[i][1] for i in range(len(df))])
    p_list=list(p1|p2)
    
    df['prob_same']=[probs[1] for probs in model.predict_proba(df[['x1', 'x2', 'x3', 'x4', 'x5']].values)]
    sim_df=pd.DataFrame(index=p_list,columns=p_list,data=np.eye(len(p_list)))
    for index,row in df.iterrows():
        sim_df.loc[index[0]][index[1]]=row['prob_same']
        sim_df.loc[index[1]][index[0]]=row['prob_same']
    
    # 聚类
    from sklearn.cluster import AffinityPropagation
    af = AffinityPropagation(affinity='precomputed').fit(sim_df.values)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
        
    p_array=np.array(p_list)
    cluster_res=[]
    for label_index in set(labels):
        cluster_res.append(list(p_array[labels==label_index]))
    
    f1=pairwise_f1({author:tag[author]},{author:cluster_res})