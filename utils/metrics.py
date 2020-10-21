from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import math

def cal_auc(labels, preds):
    auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
    return auc

def get_idcg(truth, topk):
    truth_list = []
    idcg = 0.0
    for i in range(len(truth)):
        if truth[i] == 1:
            truth_list.append(1)
    if len(truth_list) <= topk:
        for i in range(len(truth_list)):
            idcg = idcg + math.log(2) / math.log(i + 2)
    else:
        for i in range(topk):
            idcg = idcg + math.log(2) / math.log(i + 2)
    if idcg == 0:
        idcg = 1.0
        print("zero error")
    return idcg

def cal_ndcg_float(truth, preds, topk):
    truth_pred = []
    for i in range(len(truth)):
        truth_pred.append([truth[i], preds[i]])
    truth_pred_sorted = sorted(truth_pred, key = lambda e:e[1],reverse = True)
    dcg = 0.0
    idcg = get_idcg(truth, topk)
    if len(truth) <= topk:
        for k in range(len(truth)):
            if truth_pred_sorted[k][0] == 1:
                dcg = dcg + math.log(2) / math.log(k + 2)
    else:
        for k in range(topk):
            if truth_pred_sorted[k][0] == 1:
                dcg = dcg + math.log(2) / math.log(k + 2)
    ndcg = dcg/idcg
    return ndcg

def evaulate(y_pred, truth, test_data, task):
    if task == "user2item":
        score = roc_auc_score(truth, y_pred)
        sess_id = test_data['session_id']

        ndcg = []
        truth_i = []
        pred_i = []
        for i in range(len(truth)):
            truth_i.append(truth[i])
            pred_i.append(y_pred[i])
            if i + 1 == len(truth):
                ndcg.append(cal_ndcg_float(truth_i, pred_i, 10))
                truth_i = []
                pred_i = []
            elif sess_id[i] != sess_id[i + 1]:
                ndcg.append(cal_ndcg_float(truth_i, pred_i, 10))
                truth_i = []
                pred_i = []
        ndcg = np.mean(np.array(ndcg))
        print("auc score:" + str(score))
        print("ndcg score:" + str(ndcg))
        return score
    elif task == "item2item":
        score = cal_auc(truth, y_pred)
        print("auc score:" + str(score))
        return score
    elif task == "vert_classify":
        y_class = []
        for list_i in y_pred:
            list_a = list_i.tolist()
            y_class.append(list_a.index(max(list_a)))
        acc = accuracy_score(truth, y_class)
        f1 = f1_score(truth, y_class, average='macro')
        print('ACC:%.6f F1:%.6f' % (acc, f1))
        return f1
    elif task == "pop_predict":
        y_class = []
        for list_i in y_pred:
            list_a = list_i.tolist()
            y_class.append(list_a.index(max(list_a)))
        truth = test_data['label']
        acc = accuracy_score(truth, y_class)
        f1 = f1_score(truth, y_class, average='macro')
        print('ACC:%.6f F1:%.6f' % (acc, f1))
        return f1
    elif task == "local_news":
        auc = roc_auc_score(truth, y_pred)
        for i in range(len(y_pred)):
            if y_pred[i] > 0.2:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        acc = accuracy_score(truth, y_pred)
        f1 = f1_score(truth, y_pred, average='macro')
        print('ACC:%.6f AUC:%.6f F1:%.6f' % (acc, auc, f1))
        return f1
    else:
        print("task error")
        return 0