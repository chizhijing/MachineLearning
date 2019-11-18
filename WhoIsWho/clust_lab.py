from graph import get_train_data,GraphAuthors,pairwise_f1,GraphTitles,is_same_org
from common import pairwise_f1_new,process_org2


if __name__=='__main__':
    author_list=['li_guo', 'bo_shen', 'di_wang', 'long_wang', 'qiang_xu', 
             'xiang_wang', 'changming_liu', 'kenji_kaneko', 'guohua_chen', 'hai_jin', 
             'jia_li', 'guoliang_li', 'lan_wang', 'alessandro_giuliani', 'jiang_he', 
             'xiang_gao', 'jianping_wu', 'peng_shi', 'feng_wu', 'jing_zhu']
    train_data,train_tag=get_train_data(author_list)
    graph_author_dict={}
    graph_author_dict3={}
    graph_title_dict={}

# 使用authors信息构建的图结构
    for author_select in author_list:
        print(author_select)
        g_a=GraphAuthors(train_data[author_select],author_select)
        g_a.add_edge()
        graph_author_dict[author_select]=g_a

# 
    for author_select in author_list:
        print(author_select)
        g_a=GraphAuthors(train_data[author_select],author_select)
        g_a.add_edge3()
        graph_author_dict3[author_select]=g_a

    for author_select in author_list:
        print(author_select)
        g_t=GraphTitles(train_data[author_select])
        g_t.set_init_graph(graph_author_dict[author_select].graph)
        g_t.get_dist_matrix(num_topic=20)
        g_t.del_edge(dist_del=0.99)
        g_t.get_origin_cluster_dist(dist_type=2)
        graph_title_dict[author_select]=g_t


    for author_select in author_list:
        real_res = {author_select: train_tag[author_select]}
        model_author = {author_select: graph_author_dict[author_select].get_connected_components()}
        model_author3 = {author_select: graph_author_dict3[author_select].get_connected_components()}
#        model_title = {author_select:graph_title_dict[author_select].get_cluster2(0.3,200)}
        print(author_select, len(real_res[author_select]))
        print('----> model_author','%.2f' % pairwise_f1(real_res, model_author),
              '%.2f' % pairwise_f1_new(real_res, model_author),
              len(model_author[author_select]))
        print('----> model_author3','%.2f' % pairwise_f1(real_res, model_author3),
              '%.2f' % pairwise_f1_new(real_res, model_author3),
              len(model_author3[author_select]))
#        print('----> model_title','%.2f'%pairwise_f1(real_res,model_title),
#              '%.2f'%pairwise_f1_new(real_res,model_title),
#              len(model_title[author_select]))
#    
    for author_select in author_list:
        g_a=graph_author_dict[author_select]
        g_a.node

    for org in graph_author_dict['xiang_wang'].author_org:
        new_org=org.lower()
#        new_org=process_org2(org)
        if new_org!='':
            print(new_org)
        
            