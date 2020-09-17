import argparse
from utils.data_loader import load_data
from train import train_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default='/mnt/KRED_publish/data/', help='root path of data')

    ##turning paras
    parser.add_argument('--learning_rate', action='store_true', default=0.0001, help='learning rate')
    parser.add_argument('--epoch', action='store_true', default=100, help='epoch num')
    parser.add_argument('--batch_size', action='store_true', default=16, help='batch size')
    parser.add_argument('--l2_regular', action='store_true', default=0.00001, help='l2 regular')

    ##task specific parameter
    parser.add_argument('--training_type', action='store_true', default="single_task", help='single_task training or multi-task training')
    parser.add_argument('--task', action='store_true', default="user2item", help='task types: user2item, item2item, vert_classify, pop_predict, local_news')

    parser.add_argument('--news_entity_num', action='store_true', default=20, help='fix a news entity num to news_entity_num')
    parser.add_argument('--entity_neighbor_num', action='store_true', default=20, help='nerighbor num for a entity')
    parser.add_argument('--user_his_num', action='store_true', default=20, help='user history num')
    parser.add_argument('--negative_num', action='store_true', default=6, help='1 postive and negative_num-1 negative in training set')
    parser.add_argument('--smooth_lamda', action='store_true', default=10, help='smooth_lamda in softmax in loss function')

    parser.add_argument('--embedding_dim', action='store_true', default=90, help='embedding dim for enity_embedding dv uv')
    parser.add_argument('--layer_dim', action='store_true', default=128, help='layer dim')

    parser.add_argument('--logdir', action='store_true', default="EXP_num", help='the dir for save predict results')


    args = parser.parse_args()

    data = load_data(args)

    train_test(args, data)

if __name__ == "__main__":
    main()
