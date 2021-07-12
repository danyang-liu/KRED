from model.KRED import KREDModel
from model.KRED import Softmax_BCELoss
import torch
from torch import optim, nn
from trainer.trainer import Trainer
from base.base_data_loader import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.metrics import *
from utils.util import *

class NewsDataset(Dataset):
    def __init__(self, dic_data, transform=None):
        self.dic_data = dic_data
        self.transform = transform
    def __len__(self):
        return len(self.dic_data['label'])
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'item1': self.dic_data['item1'][idx], 'item2': self.dic_data['item2'][idx], 'label': self.dic_data['label'][idx]}
        return sample

def multi_task_training(config, data):
    user_history_dict, entity_embedding, relation_embedding, entity_adj, relation_adj, doc_feature_dict, entity_num, position_num, type_num, user2item_train, user2item_test, vert_train, vert_test, local_train, local_test, pop_train, pop_test, item2item_train, item2item_test = data
    train_data_u2i = NewsDataset(user2item_train)
    train_sampler_u2i = RandomSampler(train_data_u2i)
    train_dataloader_u2i = DataLoader(train_data_u2i, sampler=train_sampler_u2i, batch_size=config['data_loader']['batch_size'],
                                      collate_fn=my_collate_fn, pin_memory=False)

    train_data_vert = NewsDataset(vert_train)
    train_sampler_vert = RandomSampler(train_data_vert)
    train_dataloader_vert = DataLoader(train_data_vert, sampler=train_sampler_vert, batch_size=config['data_loader']['batch_size'],
                                       pin_memory=False)

    train_data_pop = NewsDataset(pop_train)
    train_sampler_pop = RandomSampler(train_data_pop)
    train_dataloader_pop = DataLoader(train_data_pop, sampler=train_sampler_pop, batch_size=config['data_loader']['batch_size'],
                                      pin_memory=False)

    train_data_i2i = NewsDataset(item2item_train)
    train_sampler_i2i = RandomSampler(train_data_i2i)
    train_dataloader_i2i = DataLoader(train_data_i2i, sampler=train_sampler_i2i, batch_size=config['data_loader']['batch_size'],
                                      pin_memory=False)

    device, deviceids = prepare_device(config['n_gpu'])

    model = KREDModel(config, user_history_dict, doc_feature_dict, entity_embedding, relation_embedding, entity_adj,
                      relation_adj, entity_num, position_num, type_num).cuda()

    pretrain_epoch = 0
    while (pretrain_epoch < 5):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=0)
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

        criterion = Softmax_BCELoss(config)
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

        optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=args.l2_regular)
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

    if config['trainer']['task'] == "user2item":
        criterion = Softmax_BCELoss(config)
        train_data_loader = train_dataloader_u2i
    elif config['trainer']['task'] == "item2item":
        criterion = Softmax_BCELoss(config)
        train_data_loader = train_dataloader_i2i
    elif config['trainer']['task'] == "vert_classify":
        criterion = nn.CrossEntropyLoss()
        train_data_loader = train_dataloader_vert
    elif config['trainer']['task'] == "pop_predict":
        criterion = nn.CrossEntropyLoss()
        train_data_loader = train_dataloader_pop
    else:
        print("Error: task name error.")

    trainer = Trainer(config, model, criterion, optimizer, device, train_data_loader, data[-1])
    trainer.train()


def single_task_training(config, data):
    user_history_dict, entity_embedding, relation_embedding, entity_adj, relation_adj, doc_feature_dict, entity_num, position_num, type_num, train_data, test_data = data

    if config['trainer']['task'] == "user2item":
        train_data_u2i = NewsDataset(train_data)
        train_sampler_u2i = RandomSampler(train_data_u2i)
        train_dataloader_u2i = DataLoader(train_data_u2i, sampler=train_sampler_u2i,
                                          batch_size=config['data_loader']['batch_size'],
                                          collate_fn=my_collate_fn, pin_memory=False)
        criterion = Softmax_BCELoss(config)
        train_data_loader = train_dataloader_u2i
    elif config['trainer']['task'] == "item2item":
        train_data_i2i = NewsDataset(train_data)
        train_sampler_i2i = RandomSampler(train_data_i2i)
        train_dataloader_i2i = DataLoader(train_data_i2i, sampler=train_sampler_i2i,
                                          batch_size=config['data_loader']['batch_size'],
                                          pin_memory=False)
        criterion = Softmax_BCELoss(config)
        train_data_loader = train_dataloader_i2i
    elif config['trainer']['task'] == "vert_classify":
        train_data_vert = NewsDataset(train_data)
        train_sampler_vert = RandomSampler(train_data_vert)
        train_dataloader_vert = DataLoader(train_data_vert, sampler=train_sampler_vert,
                                           batch_size=config['data_loader']['batch_size'],
                                           pin_memory=False)
        criterion = nn.CrossEntropyLoss()
        train_data_loader = train_dataloader_vert
    elif config['trainer']['task'] == "pop_predict":
        train_data_pop = NewsDataset(train_data)
        train_sampler_pop = RandomSampler(train_data_pop)
        train_dataloader_pop = DataLoader(train_data_pop, sampler=train_sampler_pop,
                                          batch_size=config['data_loader']['batch_size'],
                                          pin_memory=False)
        criterion = nn.CrossEntropyLoss()
        train_data_loader = train_dataloader_pop
    else:
        print("Error: task name error.")

    device, deviceids = prepare_device(config['n_gpu'])

    model = KREDModel(config, user_history_dict, doc_feature_dict, entity_embedding, relation_embedding, entity_adj,
                      relation_adj, entity_num, position_num, type_num).cuda()

    optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=0)

    trainer = Trainer(config, model, criterion, optimizer, device, train_data_loader, data[-1])
    trainer.train()


def testing(test_data, config):
    if config['trainer']['task'] == "user2item":
        task_index = 0
    elif config['trainer']['task'] == "item2item":
        task_index = 4
    elif config['trainer']['task'] == "vert_classify":
        task_index = 1
    elif config['trainer']['task'] == "pop_predict":
        task_index = 3
    model = torch.load('./out/saved/models/KRED/checkpoint.pt')
    model.eval()
    y_pred = []
    start_list = list(range(0, len(test_data['label']), config['data_loader']['batch_size']))
    for start in start_list:
        if start + config['data_loader']['batch_size'] <= len(test_data['label']):
            end = start + config['data_loader']['batch_size']
        else:
            end = len(test_data['label'])
        # had to change 'user_id' to 'item1' and 'news_id' to 'item2' according to key declarations in utils.util load_data_mind function
        out = model(test_data['item1'][start:end], test_data['item2'][start:end], config['data_loader']['batch_size'])[
            task_index].cpu().data.numpy()

        y_pred.extend(out)
    truth = test_data['label']
    score = evaluate(y_pred, truth, test_data, config['trainer']['task'])




