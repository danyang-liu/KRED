import json
import torch
import random
import numpy as np
import math
import os
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import requests
import math
import zipfile
#from logger.logger import *
from tqdm import tqdm


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

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

def my_collate_fn(batch):
    return batch

def construct_entity_dict(entity_file):
    fp_entity2id = open(entity_file, 'r', encoding='utf-8')
    entity_dict = {}
    entity_num_all = int(fp_entity2id.readline().split('\n')[0])
    lines = fp_entity2id.readlines()
    for line in lines:
        entity, entityid = line.strip().split('\t')
        entity_dict[entity] = entityid
    return entity_dict

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

def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath

def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)

def get_mind_data_set(type):
    """ Get MIND dataset address

    Args:
        type (str): type of mind dataset, must be in ['large', 'small', 'demo']

    Returns:
        list: data url and train valid dataset name
    """
    assert type in ["large", "small", "demo"]

    if type == "large":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDlarge_train.zip",
            "MINDlarge_dev.zip",
            "MINDlarge_utils.zip",
        )

    elif type == "small":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDsmall_train.zip",
            "MINDsmall_dev.zip",
            "MINDsma_utils.zip",
        )

    elif type == "demo":
        return (
            "https://recodatasets.blob.core.windows.net/newsrec/",
            "MINDdemo_train.zip",
            "MINDdemo_dev.zip",
            "MINDdemo_utils.zip",
        )

def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        azure_container_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))

def get_user2item_data(config):
    negative_num = config['trainer']['train_neg_num']
    train_data = {}
    user_id = []
    news_id = []
    label = []
    fp_train = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        positive_list = []
        negative_list = []
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                positive_list.append(newsid)
            else:
                negative_list.append(newsid)
        for pos_news in positive_list:
            user_id.append(userid+ "_train")
            if len(negative_list) >= negative_num:
                neg_news = random.sample(negative_list, negative_num)
            else:
                neg_news = negative_list
                for i in range(negative_num - len(negative_list)):
                    neg_news.append("N0")
            all_news = neg_news
            all_news.append(pos_news)
            news_id.append(all_news)
            label.append([])
            for i in range(negative_num):
                label[-1].append(0)
            label[-1].append(1)

    train_data['item1'] = user_id
    train_data['item2'] = news_id
    train_data['label'] = label

    dev_data = {}
    session_id = []
    user_id = []
    news_id = []
    label = []
    fp_dev = open(config['data']['valid_behavior'], 'r', encoding='utf-8')
    for line in fp_dev:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            session_id.append(index)
            user_id.append(userid+ "_dev")
            if news_label == "1":
                news_id.append(newsid)
                label.append(1)
            else:
                news_id.append(newsid)
                label.append(0)

    dev_data['item1'] = user_id
    dev_data['session_id'] = session_id
    dev_data['item2'] = news_id
    dev_data['label'] = label

    return train_data, dev_data

def build_user_history(config):
    user_history_dict = {}
    fp_train_behavior = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    for line in fp_train_behavior:
        index, user_id, imp_time, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id+"_train"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_train"] = history.split(' ')
            for i in range(config['model']['user_his_num']-len(history.split(' '))):
                user_history_dict[user_id + "_train"].append("N0")
    fp_dev_behavior = open(config['data']['valid_behavior'], 'r', encoding='utf-8')
    for line in fp_dev_behavior:
        index, user_id, imp_time, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id+"_dev"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_dev"] = history.split(' ')
            for i in range(config['model']['user_his_num']-len(history.split(' '))):
                user_history_dict[user_id + "_dev"].append("N0")
    return user_history_dict

def build_news_features_mind(config):
    entity2id_dict = {}
    fp_entity2id = open(config['data']['entity_index'], 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])
    for line in fp_entity2id.readlines():
        entity, entityid = line.strip().split('\t')
        entity2id_dict[entity] = int(entityid) + 1

    news_features = {}

    news_feature_dict = {}
    fp_train_news = open(config['data']['train_news'], 'r', encoding='utf-8')
    for line in fp_train_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        news_feature_dict[newsid] = (title+" "+abstract, entity_info_title, entity_info_abstract)
    # entityid, entity_freq, entity_position, entity_type
    fp_dev_news = open(config['data']['valid_news'], 'r', encoding='utf-8')
    for line in fp_dev_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        news_feature_dict[newsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)

    #deal with doc feature
    entity_type_dict = {}
    entity_type_index = 1
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    for news in news_feature_dict:
        sentence_embedding = model.encode(news_feature_dict[news][0])
        news_entity_feature_list = []
        title_entity_json = json.loads(news_feature_dict[news][1])
        abstract_entity_json = json.loads(news_feature_dict[news][2])
        news_entity_feature = {}
        for item in title_entity_json:
            if item['Type'] not in entity_type_dict:
                entity_type_dict[item['Type']] = entity_type_index
                entity_type_index = entity_type_index + 1
            news_entity_feature[item['WikidataId']] = (len(item['OccurrenceOffsets']), 1, entity_type_dict[item['Type']]) #entity_freq, entity_position, entity_type
        for item in abstract_entity_json:
            if item['WikidataId'] in news_entity_feature:
                news_entity_feature[item['WikidataId']] = (news_entity_feature[item['WikidataId']][0] + len(item['OccurrenceOffsets']), 1, entity_type_dict[item['Type']])
            else:
                if item['Type'] not in entity_type_dict:
                    entity_type_dict[item['Type']] = entity_type_index
                    entity_type_index = entity_type_index + 1
                news_entity_feature[item['WikidataId']] = (len(item['OccurrenceOffsets']), 2, entity_type_dict[
                    item['Type']])  # entity_freq, entity_position, entity_type
        for entity in news_entity_feature:
            if entity in entity2id_dict:
                news_entity_feature_list.append([entity2id_dict[entity], news_entity_feature[entity][0], news_entity_feature[entity][1], news_entity_feature[entity][2]])
        news_entity_feature_list.append([0,0,0,0])
        news_features[news] = (sentence_embedding, news_entity_feature_list)
    news_features["N0"] = (np.zeros(config['model']['document_embedding_dim']), [[0,0,0,0]])
    return news_features, 100, 10, 100

def construct_adj_mind(config):#graph is triple
    print('constructing adjacency matrix ...')
    graph_file_fp = open(config['data']['knowledge_graph'], 'r', encoding='utf-8')
    graph = []
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        graph.append([int(linesplit[0])+1, int(linesplit[2])+1, int(linesplit[1])+1])
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

    fp_entity2id = open(config['data']['entity_index'], 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])+1
    entity_adj = []
    relation_adj = []
    for i in range(entity_num):
        entity_adj.append([])
        relation_adj.append([])
    for key in kg.keys():
        for index in range(config['model']['entity_neighbor_num']):
            i = random.randint(0,len(kg[key])-1)
            entity_adj[int(key)].append(int(kg[key][i][0]))
            relation_adj[int(key)].append(int(kg[key][i][1]))

    return entity_adj, relation_adj

def construct_embedding_mind(config):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(config['data']['entity_embedding'], 'r', encoding='utf-8')
    fp_relation_embedding = open(config['data']['relation_embedding'], 'r', encoding='utf-8')
    zero_array = np.zeros(config['model']['entity_embedding_dim'])
    entity_embedding.append(zero_array)
    relation_embedding.append(zero_array)
    for line in fp_entity_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    for line in fp_relation_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        relation_embedding.append(linesplit)
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)

def build_vert_data(config):
    random.seed(2020)
    vert_label_dict = {}
    label_index = 0
    all_news_data = []
    vert_train = {}
    vert_dev = {}
    item1_list_train = []
    item2_list_train = []
    label_list_train = []
    item1_list_dev = []
    item2_list_dev = []
    label_list_dev = []
    fp_train_news = open(config['data']['train_news'], 'r', encoding='utf-8')
    for line in fp_train_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        if vert not in vert_label_dict:
            vert_label_dict[vert] = label_index
            label_index = label_index + 1
        all_news_data.append((newsid, vert_label_dict[vert]))
    print(vert_label_dict)
    for i in range(len(all_news_data)):
        if random.random()<0.8:
            item1_list_train.append("U0")
            item2_list_train.append(all_news_data[i][0])
            label_list_train.append(all_news_data[i][1])
        else:
            item1_list_dev.append("U0")
            item2_list_dev.append(all_news_data[i][0])
            label_list_dev.append(all_news_data[i][1])
    vert_train['item1'] = item1_list_train
    vert_train['item2'] = item2_list_train
    vert_train['label'] = label_list_train
    vert_dev['item1'] = item1_list_dev
    vert_dev['item2'] = item2_list_dev
    vert_dev['label'] = label_list_dev

    return vert_train, vert_dev

def build_pop_data(config):
    fp_train = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    news_imp_dict = {}
    pop_train = {}
    pop_test = {}
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [1,1]
                else:
                    news_imp_dict[newsid][0] = news_imp_dict[newsid][0] + 1
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
            else:
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [0,1]
                else:
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
    return pop_train, pop_test

def build_item2item_data(config):
    fp_train = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    item2item_train = {}
    item2item_test = {}
    news_click_dict = {}
    news_pair_dict = {}
    all_news_set = set()
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            positive_list = []
            newsid, news_label = news.split('-')
            all_news_set.add(newsid)
            if news_label == "1":
                positive_list.append(newsid)
                if newsid not in news_click_dict:
                    news_click_dict[newsid] = 1
                else:
                    news_click_dict[newsid] = news_click_dict[newsid] + 1
        news = history.split(' ')
        positive_list = []
        for newsid in news:
            positive_list.append(newsid)
            if newsid not in news_click_dict:
                news_click_dict[newsid] = 1
            else:
                news_click_dict[newsid] = news_click_dict[newsid] + 1
        if len(positive_list) >= 2:
            for i in range(len(positive_list) - 1):
                for j in range(i, len(positive_list)):
                    if (positive_list[i], positive_list[j]) not in news_pair_dict and (
                    positive_list[j], positive_list[i]) not in news_pair_dict:
                        news_pair_dict[(positive_list[i], positive_list[j])] = 1
                    elif (positive_list[i], positive_list[j]) in news_pair_dict:
                        news_pair_dict[(positive_list[i], positive_list[j])] = news_pair_dict[(
                        positive_list[i], positive_list[j])] + 1
                    else:
                        news_pair_dict[(positive_list[j], positive_list[i])] = news_pair_dict[(positive_list[j], positive_list[i])] + 1
    return item2item_train, item2item_test

def load_data_mind(config):
    user_history = build_user_history(config)

    entity_embedding, relation_embedding = construct_embedding_mind(config)

    entity_adj, relation_adj = construct_adj_mind(config)

    news_feature, max_entity_freq, max_entity_pos, max_entity_type = build_news_features_mind(config)

    if config['trainer']['training_type'] == "multi-task":
        train_data, dev_data = get_user2item_data(config)
        vert_train, vert_test = build_vert_data(config)
        pop_train, pop_test = build_pop_data(config)
        item2item_train, item2item_test = build_item2item_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data, vert_train, vert_test, pop_train, pop_test, item2item_train, item2item_test
    elif config['trainer']['task'] == "user2item":
        train_data, dev_data = get_user2item_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data
    elif config['trainer']['task'] == "item2item":
        item2item_train, item2item_test = build_item2item_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, item2item_train, item2item_test
    elif config['trainer']['task'] == "vert_classify":
        vert_train, vert_test = build_vert_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, vert_train, vert_test
    elif config['trainer']['task'] == "pop_predict":
        pop_train, pop_test = build_pop_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, pop_train, pop_test
    else:
        print("task error, please check config")




