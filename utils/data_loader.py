import torch
import random
import numpy as np

def construct_adj(graph_file, entity2id_file, args):#graph is triple
    print('constructing adjacency matrix ...')
    graph_file_fp = open(graph_file, 'r', encoding='utf-8')
    graph = []
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        if len(linesplit) > 1:
            graph.append([linesplit[0], linesplit[2], linesplit[1]])

    kg = {}
    for triple in graph:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    fp_entity2id = open(entity2id_file, 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])
    entity_adj = []
    relation_adj = []
    for i in range(entity_num):
        entity_adj.append([])
        relation_adj.append([])
    for key in kg.keys():
        for index in range(args.entity_neighbor_num):
            i = random.randint(0,len(kg[key])-1)
            entity_adj[int(key)].append(int(kg[key][i][0]))
            relation_adj[int(key)].append(int(kg[key][i][1]))

    return entity_adj, relation_adj

def construct_embedding(entity_embedding_file, relation_embedding_file):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(entity_embedding_file, 'r', encoding='utf-8')
    fp_relation_embedding = open(relation_embedding_file, 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    for line in fp_relation_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        relation_embedding.append(linesplit)
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)


def construct_user_history(user_history_file, args):
    print('constructing user history ...')
    user_history_dict = {}
    fp_user_histroy = open(user_history_file, 'r', encoding='utf-8')
    for line in fp_user_histroy:
        linesplit = line.split('\n')[0].split('\t')
        userid = linesplit[0]
        user_history = linesplit[1].split(' ')[:-1]
        if len(user_history) > args.user_his_num:
            sample_user_history = user_history[-args.user_his_num:]
        else:
            sample_user_history = []
            for index in range(len(user_history)):
                sample_user_history.append(user_history[index])
            for i in range(len(user_history),args.user_his_num):
                sample_user_history.append('Doc_0')#padding zero one
            pass
        user_history_dict[userid] = sample_user_history
    return user_history_dict

def construct_doc_feature(doc_feature_file, entity2id_file, news_entity_num):
    print('constructing doc feature ...')
    doc_feature_dict = {}
    entity_num = 0
    position_num = 0
    type_num =0
    fp_entity2id = open(entity2id_file, 'r', encoding='utf-8')
    entity_dict = {}
    entity_num = int(fp_entity2id.readline().split('\n')[0])
    lines = fp_entity2id.readlines()
    for line in lines:
        entity, entityid = line.strip().split('\t')
        entity_dict[entity] = entityid

    fp_doc_feature = open(doc_feature_file, 'r', encoding='utf-8')
    for line in fp_doc_feature:
        entityid_list = []
        entity_num_list = []
        istitle_list = []
        type_list = []
        linesplit = line.split('\n')[0].split('\t')

        for i in range(1, len(linesplit) - 1):
            if len(entityid_list) < news_entity_num:
                doc_feature = linesplit[i].split(' ')
                if doc_feature[0] in entity_dict:
                    entityid_list.append(int(entity_dict[doc_feature[0]]))
                    entity_num_list.append(int(doc_feature[1]))
                    istitle_list.append(int(doc_feature[2]))
                    type_list.append(int(doc_feature[3]))
                    if int(doc_feature[1]) > entity_num:
                        entity_num = int(doc_feature[1])
                    if int(doc_feature[2]) > position_num:
                        position_num = int(doc_feature[2])
                    if int(doc_feature[3]) > type_num:
                        type_num = int(doc_feature[3])
        if len(entityid_list) < news_entity_num:
            for i in range(len(entityid_list), news_entity_num):
                entityid_list.append(0)
                entity_num_list.append(0)
                istitle_list.append(0)
                type_list.append(0)

        context_vec = linesplit[-1].split(' ')
        context_vec = [float(vec) for vec in context_vec]
        doc_feature_dict[linesplit[0]] = (entityid_list, entity_num_list, istitle_list, type_list, context_vec)

    fp_doc_feature.close()

    return doc_feature_dict, entity_num+1, position_num+1, type_num+1

def get_vec(vec):
    vec_list = []
    vec_split1 = vec.split()
    for i in range(len(vec_split1)):
        vec_list.append(float(vec_split1[i]))
    return vec_list

def construct_train(train_file, args):
    print('constructing train ...')
    train_data = {}
    user_id = []
    news_id = []
    label = []
    fp_train = open(train_file, 'r', encoding='utf-8')
    train_index = 0
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        if train_index % args.negative_num == 0:
            user_id.append(linesplit[1])
            news_id.append([])
            label.append([])
            news_id[-1].append(linesplit[2])
            label[-1].append(float(linesplit[3]))
        else:
            news_id[-1].append(linesplit[2])
            label[-1].append(float(linesplit[3]))
        train_index = train_index + 1

    train_data['user_id'] = user_id
    train_data['news_id'] = news_id
    train_data['label'] = label
    return train_data

def construct_test(test_file):
    print('constructing test ...')
    test_data = {}
    session_id = []
    user_id = []
    news_id = []
    label = []
    fp_test = open(test_file, 'r', encoding='utf-8')
    for line in fp_test:
        linesplit = line.split('\n')[0].split('\t')
        session_id.append(linesplit[0])
        user_id.append(linesplit[1])
        news_id.append(linesplit[2])
        label.append(float(linesplit[3]))
    test_data['session_id'] = session_id
    test_data['user_id'] = user_id
    test_data['news_id'] = news_id
    test_data['label'] = label
    return test_data


def construct_multi_data(multi_data_file):
    print('constructing multi_data ...')
    multi_data = {}
    user_id = []
    news_id = []
    label = []
    UV = []
    fp_doc_feature = open(multi_data_file, 'r', encoding='utf-8')
    for line in fp_doc_feature:
        linesplit = line.split('\n')[0].split('\t')
        user_id.append("User_0")
        news_id.append(linesplit[0])
        label.append(int(linesplit[1]))

    multi_data['user_id'] = user_id
    multi_data['news_id'] = news_id
    multi_data['label'] = label
    return multi_data

def construct_item2item_train(train_file):
    print('constructing train ...')
    train_data = {}
    user_id = []
    news_id = []
    label = []

    fp_train = open(train_file, 'r', encoding='utf-8')
    train_index = 0
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        if train_index % 6 == 0:
            user_id.append(linesplit[0])
            news_id.append([])
            label.append([])
            news_id[-1].append(linesplit[1])
            label[-1].append(float(linesplit[2]))
        else:
            news_id[-1].append(linesplit[1])
            label[-1].append(float(linesplit[2]))
        train_index = train_index + 1

    train_data['user_id'] = user_id
    train_data['news_id'] = news_id
    train_data['label'] = label
    return train_data

def construct_item2item_test(test_file):
    test_data = {}
    user_id = []
    news_id = []
    label = []
    fp_train = open(test_file, 'r', encoding='utf-8')
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        user_id.append(linesplit[0])
        news_id.append(linesplit[1])
        label.append(float(linesplit[2]))
    test_data['user_id'] = user_id
    test_data['news_id'] = news_id
    test_data['label'] = label
    return test_data

def my_collate_fn(batch):
    return batch

def real_batch(batch):
    data = {}
    data['item1'] = []
    data['item2'] = []
    data['label'] = []
    for item in batch:
        data['item1'].append(item['item1'])
        data['item2'].append(item['item2'])
        data['label'].append(item['label'])
    return data


def load_data(args):

    graph_file = args.rootpath+"news_graph/triple2id.txt"
    entity2id_file = args.rootpath+"news_graph/entity2id.txt"
    entity_adj, relation_adj = construct_adj(graph_file, entity2id_file, args)

    entity_embedding_file = args.rootpath+ "news_graph/entity2vec.vec"
    relation_embedding_file = args.rootpath + "news_graph/relation2vec.vec"
    entity_embedding, relation_embedding = construct_embedding(entity_embedding_file, relation_embedding_file)

    user_history_file = args.rootpath + "user_history.tsv"
    user_history_dict = construct_user_history(user_history_file, args)

    train_data = construct_train(args.rootpath +"u2i_train.tsv", args)
    test_data = construct_test(args.rootpath  + "u2i_test.tsv")

    doc_feature_file = args.rootpath + "doc_feature_train_test_new.tsv"
    doc_feature_dict, entity_num, position_num, type_num = construct_doc_feature(doc_feature_file, entity2id_file, args.news_entity_num)

    vert_train = construct_multi_data(args.rootpath + "vert_train.tsv")
    vert_test = construct_multi_data(args.rootpath + "vert_test.tsv")
    local_train = construct_multi_data(args.rootpath + "local_train.tsv")
    local_test = construct_multi_data(args.rootpath + "local_test.tsv")
    pop_train = construct_multi_data(args.rootpath + "pop_train.tsv")
    pop_test = construct_multi_data(args.rootpath + "pop_test.tsv")
    item2item_train_data = construct_item2item_train(args.rootpath + "i2i_train.tsv")
    item2item_test_data = construct_item2item_test(args.rootpath + "i2i_test.tsv")


    print('constructing data finishced ...')

    return user_history_dict, entity_embedding, relation_embedding, entity_adj, relation_adj, doc_feature_dict, entity_num, position_num, type_num, train_data, test_data, vert_train, vert_test, local_train, local_test, pop_train, pop_test, item2item_train_data, item2item_test_data
    #return train_data, test_data, vert_train, vert_test, local_train, local_test, pop_train, pop_test, item2item_train_data, item2item_test_data