from src.KRED import KRED
from src.KRED import Softmax_BCELoss
import torch
from torch import optim, nn
import os
from utils.data_loader import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.pytorchtools import EarlyStopping
from utils.metrics import *

class NewsDataset(Dataset):
    def __init__(self, dic_data, transform=None):
        self.dic_data = dic_data
        self.transform = transform
    def __len__(self):
        return len(self.dic_data['label'])
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'item1': self.dic_data['user_id'][idx], 'item2': self.dic_data['news_id'][idx], 'label': self.dic_data['label'][idx]}
        return sample

def train_test(args, data):
    user_history_dict, entity_embedding, relation_embedding, entity_adj, relation_adj, doc_feature_dict, entity_num, position_num, type_num, user2item_train, user2item_test, vert_train, vert_test, local_train, local_test, pop_train, pop_test, item2item_train, item2item_test  = data
    #user2item_train, user2item_test, vert_train, vert_test, local_train, local_test, pop_train, pop_test, item2item_train, item2item_test = data

    train_data_u2i = NewsDataset(user2item_train)
    train_sampler_u2i = RandomSampler(train_data_u2i)
    train_dataloader_u2i = DataLoader(train_data_u2i, sampler=train_sampler_u2i, batch_size=args.batch_size, collate_fn=my_collate_fn, pin_memory=False)

    train_data_vert = NewsDataset(vert_train)
    train_sampler_vert = RandomSampler(train_data_vert)
    train_dataloader_vert = DataLoader(train_data_vert, sampler=train_sampler_vert, batch_size=args.batch_size, pin_memory=False)

    train_data_local = NewsDataset(local_train)
    train_sampler_local = RandomSampler(train_data_local)
    train_dataloader_local = DataLoader(train_data_local, sampler=train_sampler_local, batch_size=args.batch_size, pin_memory=False)

    train_data_pop = NewsDataset(pop_train)
    train_sampler_pop = RandomSampler(train_data_pop)
    train_dataloader_pop = DataLoader(train_data_pop, sampler=train_sampler_pop, batch_size=args.batch_size, pin_memory=False)

    train_data_i2i = NewsDataset(item2item_train)
    train_sampler_i2i = RandomSampler(train_data_i2i)
    train_dataloader_i2i = DataLoader(train_data_i2i, sampler=train_sampler_i2i, batch_size=args.batch_size, pin_memory=False)

    valid_scores = []
    early_stopping = EarlyStopping(patience=2, verbose=True)

    print("learning rate {} l2_regular {}".format(args.learning_rate, args.l2_regular))

    model = KRED(args, user_history_dict, doc_feature_dict, entity_embedding, relation_embedding, entity_adj,
                 relation_adj, entity_num, position_num, type_num).cuda()

    if args.training_type == "multi-task":
        pretrain_epoch = 0
        while(pretrain_epoch < 5):
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
            total_loss_vert = 0
            model.train()
            for step, batch in enumerate(train_dataloader_vert):
                out = model(batch['item1'], batch['item2'], "vert_classify")[1]
                loss = criterion(out, torch.tensor(batch['label']).cuda())
                total_loss_vert = total_loss_vert + loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {} loss {}'.format(pretrain_epoch, total_loss_vert))

            total_loss_local = 0
            model.train()
            for step, batch in enumerate(train_dataloader_local):
                out = model(batch['item1'], batch['item2'], "local_news")[2]
                loss = criterion(out, torch.tensor(batch['label']).cuda())
                total_loss_local = total_loss_local + loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {} loss {}'.format(pretrain_epoch, total_loss_local))

            total_loss_pop = 0
            model.train()
            for step, batch in enumerate(train_dataloader_pop):
                out = model(batch['item1'], batch['item2'], "pop_predict")[3]
                loss = criterion(out, torch.tensor(batch['label']).cuda())
                total_loss_pop = total_loss_pop + loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {} loss {}'.format(pretrain_epoch, total_loss_pop))

            criterion = Softmax_BCELoss(args)
            total_loss_i2i = 0
            model.train()
            for step, batch in enumerate(train_dataloader_i2i):
                out = model(batch['item1'], batch['item2'], "item2item")[4]
                loss = criterion(out, torch.stack(batch['label']).float().cuda())
                total_loss_i2i = total_loss_i2i + loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {} loss {}'.format(pretrain_epoch, total_loss_i2i))

            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_regular)
            total_loss_u2i = 0
            model.train()
            for step, batch in enumerate(train_dataloader_u2i):
                batch = real_batch(batch)
                out = model(batch['item1'], batch['item2'], "user2item")[0]
                loss = criterion(out, torch.tensor(batch['label']).cuda())
                total_loss_u2i = total_loss_u2i + loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch {} loss {}'.format(pretrain_epoch, total_loss_u2i))
            pretrain_epoch = pretrain_epoch + 1

    for epoch in range(args.epoch):
        if args.task == "user2item":
            test_data = user2item_test
            criterion = Softmax_BCELoss(args)
            train_data_loader = train_dataloader_u2i
            task_index = 0
        elif args.task == "item2item":
            test_data = user2item_test
            criterion = nn.CrossEntropyLoss(args)
            train_data_loader = train_dataloader_i2i
            task_index = 4
        elif args.task == "vert_classify":
            test_data = user2item_test
            criterion = nn.CrossEntropyLoss(args)
            train_data_loader = train_dataloader_vert
            task_index = 1
        elif args.task == "pop_predict":
            test_data = user2item_test
            criterion = nn.CrossEntropyLoss(args)
            train_data_loader = train_dataloader_pop
            task_index = 2
        elif args.task == "local_news":
            test_data = user2item_test
            criterion = nn.CrossEntropyLoss(args)
            train_data_loader = train_dataloader_local
            task_index = 3
        else:
            print("Error: task name error.")
            break


        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_regular)
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_data_loader):
            if task_index == 0:
                batch = real_batch(batch)
            out = model(batch['item1'], batch['item2'], args.task)[task_index]
            loss = criterion(out, torch.tensor(batch['label']).cuda())
            total_loss = total_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch {} loss {}'.format(epoch, total_loss))

        model.eval()
        y_pred = []
        start_list = list(range(0, len(test_data['label']), args.batch_size))
        for start in start_list:
            if start + args.batch_size <= len(test_data['label']):
                end = start + args.batch_size
            else:
                end = len(test_data['label'])
            out = model(test_data['user_id'][start:end], test_data['news_id'][start:end], args.task)[task_index].view(end-start).cpu().data.numpy()
            y_pred = y_pred + out.tolist()
        truth = test_data['label']
        score = evaulate(y_pred, truth, test_data, args.task)
        valid_scores.append(score)

        early_stopping(score, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    model.load_state_dict(torch.load('checkpoint.pt'))
    y_pred = []
    start_list = list(range(0, len(test_data['label']), args.batch_size))
    for start in start_list:
        if start + args.batch_size <= len(test_data['label']):
            end = start + args.batch_size
        else:
            end = len(test_data['label'])
        out = model(test_data['user_id'][start:end], test_data['news_id'][start:end], args.task)[task_index].view(end - start).cpu().data.numpy()
        y_pred = y_pred + out.tolist()

    result_path = "./result_log/" + args.logdir + '/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_file_path = result_path + "predict_result.txt"
    fp = open(result_file_path, 'w')
    for line_index in range(len(y_pred)):
        fp.write(str(y_pred[line_index]) + '\t' + str(truth[line_index]) + '\n')






