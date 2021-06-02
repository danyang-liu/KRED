import sys
import os
import pickle
from train_test import *
sys.path.append('')

import argparse
from parse_config import ConfigParser



parser = argparse.ArgumentParser(description='KRED')


parser.add_argument('-c', '--config', default="./config.json", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)

from utils.util import *
data = load_data_mind(config)

#vert_train, vert_dev = build_vert_data("./data/mind_small_train/news.tsv")
#build_pop_data("./data/mind_small_train/behaviors.tsv")
#build_item2item_data("./data/mind_small_train/behaviors.tsv")
#
# user_history = build_user_history(config, "./data/mind_small_train/behaviors.tsv", "./data/mind_small_dev/behaviors.tsv")
#
# entity_embedding, relation_embedding = construct_embedding(config, "./data/wikidata-graph/entity2vecd100.vec", "./data/wikidata-graph/relation2vecd100.vec")
#
# entity_adj, relation_adj = construct_adj(config, "./data/wikidata-graph/triple2id.txt", "./data/wikidata-graph/entity2id.txt")
#
# train_data, dev_data = get_mind_data(config, "./data/mind_small_train/behaviors.tsv", "./data/mind_small_dev/behaviors.tsv")
#
# news_feature, max_entity_freq, max_entity_pos, max_entity_type = build_news_features_mind(config, "./data/mind_small_train/news.tsv", "./data/mind_small_dev/news.tsv", "./data/wikidata-graph/entity2id.txt")
#
#
# data = user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, train_data, dev_data, news_feature,  max_entity_freq, max_entity_pos, max_entity_type
# with open("./data/data_mind_small.pkl", 'wb') as f:
#     pickle.dump(data, f)

# with open("./data/data_mind_small.pkl", 'rb') as f:
#     data = pickle.load(f)
#
# train_test(data, config)