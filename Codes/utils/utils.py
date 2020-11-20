'''
Author: Pt
Date: 2020-11-10 00:06:47
LastEditTime: 2020-11-20 10:14:28
'''
import random
import re
import json
import pickle
import torch
import os
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,log_loss,mean_squared_error,accuracy_score,f1_score
from torchtext.data.utils import get_tokenizer
from torchtext.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def word_tokenize_vocab(sent,vocab):
    """ Split sentence into wordID list using regex and vocabulary
    Args:
        sent (str): Input sentence
        vocab : vocabulary

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return [vocab[x] for x in pat.findall(sent.lower())]
    else:
        return []

def newsample(news, ratio):
    """ Sample ratio samples from news list. 
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number
    
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def news_token_generator(news_file,tokenizer,attrs):
    ''' merge and deduplicate training news and testing news then iterate, collect attrs into a string and generate it
       
    Args: 
        tokenizer: torchtext.data.utils.tokenizer
        attrs: list of attrs to be collected and yielded
    Returns: 
        a generator over attrs in news
    '''    

    # news_df_train = pd.read_table(news_file_train,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    # news_df_test = pd.read_table(news_file_test,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    # news_df = pd.concat([news_df_train,news_df_test]).drop_duplicates()

    news_df = pd.read_table(news_file,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    news_iterator = news_df.iterrows()

    for _,i in news_iterator:
        content = []
        for attr in attrs:
            content.append(i[attr])
        
        yield tokenizer(' '.join(content))

def news_token_generator_group(news_file,tokenizer,vocab,mode):
    ''' iterate news_file, collect attrs and generate them respectively
       
    Args: 
        tokenizer: torchtext.data.utils.tokenizer
        mode: int defining how many attributes to be generated
    Returns: 
        generates wordID vector of each attrs, gathered into a list
    '''
    news_df = pd.read_table(news_file,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    news_iterator = news_df.iterrows()
    
    attrs = ['title','category','subcategory','abstract']
    for _,i in news_iterator:
        result = []
        indicator = 0
        while indicator < mode:
            result.append([vocab[x] for x in tokenizer(i[attrs[indicator]])])
            indicator += 1
        yield result 

def constructVocab(news_file,attrs,save_path):
    """
        Build field using torchtext for tokenization
    
    Returns:
        torchtext.vocabulary 
    """    
    
    tokenizer = get_tokenizer('basic_english')

    vocab = build_vocab_from_iterator(news_token_generator(news_file,tokenizer,attrs))

    output = open(save_path,'wb')
    pickle.dump(vocab,output)
    output.close()

def constructNid2idx(news_file,dic_file):
    """
        Construct news to newsID dictionary, index starting from 1
    """
    f = open(news_file,'r',encoding='utf-8')
    nid2index = {}
    for line in f:
        nid,_,_,_,_,_,_,_ = line.strip("\n").split('\t')

        if nid in nid2index:
            continue

        nid2index[nid] = len(nid2index) + 1
    f.close()
    
    g = open(dic_file,'w',encoding='utf-8')
    json.dump(nid2index,g,ensure_ascii=False)
    g.close()

def constructUid2idx(behaviors_file,dic_file):
    """
        Construct user to userID dictionary, index starting from 1
    """
    f = open(behaviors_file,'r',encoding='utf-8')
    uid2index = {}
    for line in f:
        _,uid,_,_,_ = line.strip("\n").split('\t')

        if uid in uid2index:
            continue

        uid2index[uid] = len(uid2index) + 1
    f.close()
    
    g = open(dic_file,'w',encoding='utf-8')
    json.dump(uid2index,g,ensure_ascii=False)
    g.close()

def constructBasicDict(news_file,behaviors_file,data_mode,model_mode,attrs):
    """ construct basic dictionary

        Args:
        news_file: path of news file
        behavior_file: path of behavior file
        mode: [small/large]
    """    
    dict_path_v = 'data/vocab_{}_{}_{}.pkl'.format(data_mode,model_mode,'_'.join(attrs))
    dict_path_n = 'data/nid2idx_{}_{}.json'.format(data_mode,model_mode)
    dict_path_u = 'data/uid2idx_{}_{}.json'.format(data_mode,model_mode)
    
    constructVocab(news_file,attrs,dict_path_v)
    constructNid2idx(news_file,dict_path_n)
    constructUid2idx(behaviors_file,dict_path_u)

def tailorData(tsvFile, num):
    ''' tailor num rows of tsvFile to create demo data file
    
    Args: 
        tsvFile: str of data path
    Returns: 
        create tailored data file
    '''
    mode = re.search('D:/Data/NR_data/MINDsmall_(.*)/(.*).tsv',tsvFile).group(1)
    target_file = re.search('D:/Data/NR_data/MINDsmall_(.*)/(.*).tsv',tsvFile).group(2)

    target_file = 'D:/Data/NR_data/dev/'+target_file+'_{}.tsv'.format(mode)
    
    f = open(target_file,'w',encoding='utf-8')   
    count = 0
    with open(tsvFile,'r',encoding='utf-8') as g:
        for line in g:
            if count >= num:
                f.close()
                return
            f.write(line)
            count += 1

def getId2idx(file):
    """
        get Id2idx dictionary from json file 
    """
    g = open(file,'r',encoding='utf-8')
    dic = json.load(g)
    g.close()
    return dic

def getVocab(file):
    """
        get Vocabulary from pkl file
    """
    g = open(file,'rb')
    dic = pickle.load(g)
    g.close()
    return dic

def getLoss(model):
    """
        get loss function for model
    """
    if model.npratio > 0:
        loss = nn.NLLLoss()
    else:
        loss = nn.BCELoss()
    
    return loss

def getLabel(model,x):
    """
        parse labels to label indexes 
    """
    if model.npratio > 0:
        index = torch.arange(0,model.npratio + 1,device=model.device).expand(model.batch_size,-1)
        label = x['labels']==1
        label = index[label]
    else:
        label = x['labels']
    
    return label

def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    # descending rank prediction score, get corresponding index of candidate news
    order = np.argsort(y_score)[::-1]
    # get ground truth for these indexes
    y_true = np.take(y_true, order)
    # check whether the prediction news with max score is the one being clicked
    # calculate the inverse of its index
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def _cal_metric(imp_indexes, labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels),np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            result = []
            # group_auc = np.mean(
            #     [
            #         roc_auc_score(each_labels, each_preds)
            #         for each_labels, each_preds in zip(labels, preds)
            #     ]
            # )
            
            for each_imprs, each_labels, each_preds in zip(imp_indexes, labels, preds):
                try:
                    result.append(roc_auc_score(each_labels,each_preds))
                except:
                    print("error in impression:{}, labels of which is {}, predictions of which are {}".format(each_imprs,each_labels,each_preds))
                    continue
            
            group_auc = np.mean(result)    
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res

def group_labels(impression_ids, labels, preds):
    """ Devide labels and preds into several group acscording to impression_ids

    Args:
        labels (list of batch_size): ground truth label list.
        preds (list of batch_size): prediction score list.
        impression_ids (list of batch_size): group key list.

    Returns:
        all_labels: labels after group.
        all_preds: preds after group.

    """

    all_keys = list(set(impression_ids))
    all_keys.sort()
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, impression_ids):
        group_labels[k].append(l)
        group_preds[k].append(p)

    all_labels = []
    all_preds = []

    for k in all_keys:
        all_labels.append(group_labels[k])
        if len(group_labels[k]) == 1:
            print(k,group_labels[k])
            
        all_preds.append(group_preds[k])

    return all_keys, all_labels, all_preds

def _eval(model,test_iterator,interval):
    """ making prediction and gather results into groups according to impression_id, display processing every interval batches

    Args:
        model

    Returns:
        impression_id: impression ids after group
        labels: labels after group.
        preds: preds after group.

    """
    preds = []
    labels = []
    imp_indexes = []
    # test = test_iterator.load_data_from_file()
    tqdm_ = tqdm(enumerate(test_iterator))
    
    for i,batch_data_input in tqdm_:
        
        preds.extend(model.forward(batch_data_input).tolist())            
        labels.extend(batch_data_input['labels'].squeeze().tolist())
        imp_indexes.extend(batch_data_input['impression_index'].tolist())
    
    
    impr_indexes, labels, preds = group_labels(
        imp_indexes,labels, preds
    )
    
    return impr_indexes, labels, preds

def run_eval(model,test_iterator,interval=100):
    """Evaluate the given file and returns some evaluation metrics.
    
    Args:
        filename (str): A file name that will be evaluated.

    Returns:
        dict: A dictionary contains evaluation metrics.
    """
    imp_indexes, group_labels, group_preds = _eval(model,test_iterator,interval)
    res = _cal_metric(imp_indexes,group_labels,group_preds,model.metrics.split(','))
    return res

def run_train(model, iterator, optimizer, loss_func, epochs, interval=100):
    ''' train model and print loss meanwhile
    Args: 
        model: the model to be trained
        iterator: generator which provides data
        optimizer: optimizer for training
        loss_func: loss function for training
        epochs: (int) number of epochs
        interval: (int) within each epoch, the interval of training steps to display loss
    Returns: 
        model: trained model
    '''
    for epoch in range(epochs):
        tqdm_ = tqdm(enumerate(iterator))
        step = 0
        epoch_loss = 0

        for step,x in tqdm_:
            pred = model(x)
            label = getLabel(model,x)
            loss = loss_func(pred,label)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % interval == 0:
                tqdm_.set_description(
                    "epoch {:d} , step {:d} , total_loss: {:.4f}, batch_loss: {:.4f}".format(epoch, step, epoch_loss / step, loss))
    
    return model