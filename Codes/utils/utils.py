'''
Author: Pt
Date: 2020-11-10 00:06:47
LastEditTime: 2020-11-20 10:14:28
'''
import random
import re
import os
import json
import pickle
import torch
import argparse
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
import scipy.stats as ss
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score,log_loss,mean_squared_error,accuracy_score,f1_score
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

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
        int: count of paddings
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news)), ratio-len(news)
    else:
        return random.sample(news, ratio), 0


def news_token_generator(news_file_train,news_file_test,tokenizer,attrs,news_file_dev):
    ''' merge and deduplicate training news and testing news then iterate, collect attrs into a single sentence and generate it
       
    Args: 
        tokenizer: torchtext.data.utils.tokenizer
        attrs: list of attrs to be collected and yielded
    Returns: 
        a generator over attrs in news
    '''    

    news_df_train = pd.read_table(news_file_train,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    news_df_test = pd.read_table(news_file_test,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    if news_file_dev:
        news_df_dev = pd.read_table(news_file_dev,index_col=None,names=['newsID','category','subcategory','title','abstract','url','entity_title','entity_abstract'])
    news_df = pd.concat([news_df_train,news_df_test,news_df_dev]).drop_duplicates()

    news_iterator = news_df.iterrows()

    for _,i in news_iterator:
        content = []
        for attr in attrs:
            content.append(i[attr])
        
        yield tokenizer(' '.join(content))

def constructVocab(news_file_train,news_file_test,attrs,save_path,news_file_dev=None):
    """
        Build field using torchtext for tokenization
    
    Returns:
        torchtext.vocabulary 
    """

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(news_token_generator(news_file_train,news_file_test,tokenizer,attrs,news_file_dev))

    output = open(save_path,'wb')
    pickle.dump(vocab,output)
    output.close()

def constructNid2idx(news_file_train,news_file_test,dic_file_train,dic_file_test):
    """
        Construct news to newsID dictionary, index starting from 1
    """
    f = open(news_file_train,'r',encoding='utf-8')

    nid2index = {}
    for line in f:
        nid,_,_,_,_,_,_,_ = line.strip("\n").split('\t')

        if nid in nid2index:
            continue
        nid2index[nid] = len(nid2index) + 1
    
    f.close()
    h = open(dic_file_train,'w',encoding='utf-8')
    json.dump(nid2index,h,ensure_ascii=False)
    h.close()

    g = open(news_file_test,'r',encoding='utf-8')

    nid2index = {}
    for line in g:
        nid,_,_,_,_,_,_,_ = line.strip("\n").split('\t')

        if nid in nid2index:
            continue
        nid2index[nid] = len(nid2index) + 1
    
    f.close()
    h = open(dic_file_test,'w',encoding='utf-8')
    json.dump(nid2index,h,ensure_ascii=False)
    h.close()
    

def constructUid2idx(behaviors_file_train,behaviors_file_test,dic_file):
    """
        Construct user to userID dictionary, index starting from 1
    """
    f = open(behaviors_file_train,'r',encoding='utf-8')
    g = open(behaviors_file_test,'r',encoding='utf-8')

    uid2index = {}
    for line in f:
        _,uid,_,_,_ = line.strip("\n").split('\t')

        if uid in uid2index:
            continue
        uid2index[uid] = len(uid2index) + 1

    for line in g:
        _,uid,_,_,_ = line.strip("\n").split('\t')

        if uid in uid2index:
            continue
        uid2index[uid] = len(uid2index) + 1

    f.close()
    g.close()
    
    h = open(dic_file,'w',encoding='utf-8')
    json.dump(uid2index,h,ensure_ascii=False)
    h.close()

def constructBasicDict(news_file_pair,behaviors_file_pair,data_mode,attrs):
    """ construct basic dictionary

        Args:
        news_file_pair: tuple of paths of news file, first entry is training set, the other is testing set
        behavior_file_pair: tuple of paths of behavior file, first entry is training set, the other is testing set
        mode: [demo/small/large]
    """
    dict_path_v = 'data/dictionaries/vocab_{}_{}.pkl'.format(data_mode,'_'.join(attrs))
    dict_path_n_train = 'data/dictionaries/nid2idx_{}_train.json'.format(data_mode)
    dict_path_n_test = 'data/dictionaries/nid2idx_{}_test.json'.format(data_mode)
    dict_path_u = 'data/dictionaries/uid2idx_{}.json'.format(data_mode)
    
    constructVocab(news_file_pair[0],news_file_pair[1],attrs,dict_path_v)
    constructNid2idx(news_file_pair[0],news_file_pair[1],dict_path_n_train,dict_path_n_test)
    constructUid2idx(behaviors_file_pair[0],behaviors_file_pair[1],dict_path_u)

def tailorData(tsvFile, num):
    ''' tailor num rows of tsvFile to create demo data file
    
    Args: 
        tsvFile: str of data path
    Returns: 
        create tailored data file
    '''
    pattern = re.search('(.*)MIND(.*)_(.*)/(.*).tsv',tsvFile)

    directory = pattern.group(1)
    mode = pattern.group(3)
    target_file = pattern.group(4)

    target_file = directory + 'MINDdemo' + '_{}/'.format(mode) + target_file + '.tsv'.format(mode)
    print(target_file)
    
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
    if model.cdd_size > 1:
        loss = nn.NLLLoss()
    else:
        loss = nn.BCELoss()
    
    return loss

def getLabel(model,x):
    """
        parse labels to label indexes, used in NLLoss
    """
    if model.cdd_size > 1:
        index = torch.arange(0,model.cdd_size,device=model.device).expand(model.batch_size,-1)
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
                    print("auc computaion error in impression:{}, labels of which is {}, predictions of which are {}".format(each_imprs,each_labels,each_preds))
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
        if 1 in group_labels[k]:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        else:
            print("impression {} has been stripped because of lacking 1 label".format(k))

    return all_keys, all_labels, all_preds

def _eval(model,dataloader,interval):
    """ making prediction and gather results into groups according to impression_id, display processing every interval batches

    Args:
        model(torch.nn.Module)
        dataloader(torch.utils.data.DataLoader): provide data

    Returns:
        impression_id: impression ids after group
        labels: labels after group.
        preds: preds after group.

    """
    preds = []
    labels = []
    imp_indexes = []
    tqdm_ = tqdm(enumerate(dataloader))
    
    for i,batch_data_input in tqdm_:
        
        preds.extend(model.forward(batch_data_input).tolist())
       
        label = batch_data_input['labels'].squeeze().tolist()
        
        
        labels.extend(label)
        imp_indexes.extend(batch_data_input['impression_index'].tolist())
    
    
    impr_indexes, labels, preds = group_labels(
        imp_indexes,labels, preds
    )
    
    return impr_indexes, labels, preds

def evaluate(model,hparams,dataloader,interval=100):
    """Evaluate the given file and returns some evaluation metrics.
    
    Args:
        model(nn.Module)
        hparams(dict)
        dataloader(torch.utils.data.DataLoader): provide data
        interval(int): within each epoch, the interval of steps to display loss

    Returns:
        dict: A dictionary contains evaluation metrics.
    """
    hparam_list = ['name','scale','epochs','save_step','train_embedding','select','integrate','his_size','k','query_dim','value_dim','head_num']
    param_list = ['query_words','query_levels']
    model.eval()
    model.cdd_size = 1
    imp_indexes, group_labels, group_preds = _eval(model,dataloader,interval)
    res = _cal_metric(imp_indexes,group_labels,group_preds,model.metrics.split(','))
    print("evaluation results:{}".format(res))
    with open('performance.log','a+') as f:
        # model_name = '{}-{}_{}_epoch{}_step{}_[hs={},topk={}]:'.format(hparams['name'],hparams['select'],hparams['scale'], str(hparams['epochs']), str(hparams['save_step']), str(hparams['his_size']), str(hparams['k']))
        # f.write(model_name + '\n')
        d = {}
        for k,v in hparams.items():
            if k in hparam_list:
                d[k] = v
        for name, param in model.named_parameters():
            if name in param_list:
                d[name] = tuple(param.shape)

        f.write(str(d)+'\n')
        f.write(str(res) +'\n')
        f.write('\n')        
    return res

def run_train(model, dataloader, optimizer, loss_func, hparams, writer=None, interval=100, save_step=None, save_each_epoch=False):
    ''' train model and print loss meanwhile
    Args: 
        model(torch.nn.Module): the model to be trained
        dataloader(torch.utils.data.DataLoader): provide data
        optimizer(torch.nn.optim): optimizer for training
        loss_func(torch.nn.Loss): loss function for training
        hparams(dict): hyper parameters
        writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
        interval(int): within each epoch, the interval of training steps to display loss
        save_epoch(bool): whether to save the model after every epoch
    Returns: 
        model: trained model
    '''
    total_loss = 0
    
    for epoch in range(hparams['epochs']):
        epoch_loss = 0
        tqdm_ = tqdm(enumerate(dataloader))

        for step,x in tqdm_:
            pred = model(x)
            label = getLabel(model,x)
            loss = loss_func(pred,label)

            epoch_loss += loss
            total_loss += loss

            loss.backward()
            optimizer.step()
            
            if step % interval == 0:

                tqdm_.set_description(
                    "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                if writer:
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param, epoch * len(dataloader) + step)

                    writer.add_scalar('data_loss',
                            total_loss/(epoch * len(dataloader) + step),
                            epoch * len(dataloader) + step)
            optimizer.zero_grad()
            
            if save_step:
                if step % save_step == 0 and step > 0:
                    if hparams['select']:
                        save_path = 'models/model_params/{}-{}_{}_epoch{}_step{}_[hs={},topk={}].model'.format(hparams['name'],hparams['select'],hparams['scale'],epoch + 1,step, str(hparams['his_size']), str(hparams['k']))
                    else:
                        save_path = 'models/model_params/{}_{}_epoch{}_step{}_[hs={},topk={}].model'.format(hparams['name'],hparams['select'],hparams['scale'],epoch + 1,step, str(hparams['his_size']), str(hparams['k']))
                    torch.save(model.state_dict(), save_path)
                    print("saved model of step {} at epoch {}".format(step, epoch+1))

        if writer:
            writer.add_scalar('epoch_loss', epoch_loss/len(dataloader), epoch)

        if save_each_epoch:
            if hparams['select']:
                save_path = 'models/model_params/{}-{}_{}_epoch{}_[hs={},topk={}].model'.format(hparams['name'],hparams['select'],hparams['scale'],epoch+1, str(hparams['his_size']), str(hparams['k']))
            else:
                save_path = 'models/model_params/{}_{}_epoch{}_[hs={},topk={}].model'.format(hparams['name'],hparams['scale'],epoch+1, str(hparams['his_size']), str(hparams['k']))
            
            torch.save(model.state_dict(), save_path)
            print("saved model of epoch {}".format(epoch+1))

    return model

def train(model, hparams, loader_train, loader_dev, loader_validate=None, tb=False, interval=100):
    """ wrap training process
    
    Args:
        model(torch.nn.Module): the model to be trained
        loader_train(torch.utils.data.DataLoader): provide data
        hparams(dict): hyper paramaters
        en: shell parameter
    """
    model.train()
    writer = None
    
    if tb:
        if hparams['select']:
            writer = SummaryWriter('data/tb/{}-{}/{}/{}/'.format(hparams['name'], hparams['select'], hparams['scale'], datetime.now().strftime("%Y%m%d-%H")))
        else:
            writer = SummaryWriter('data/tb/{}/{}/{}/'.format(hparams['name'], hparams['scale'], datetime.now().strftime("%Y%m%d-%H")))

    print("training...")
    loss_func = getLoss(model)
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    model = run_train(model,loader_train,optimizer,loss_func,hparams,writer=writer,interval=interval,save_step=hparams['save_step'],save_each_epoch=hparams['save_each_epoch'])

    # if save:
    #     ### independent of hparams['save_path']
    #     if hparams['select']:
    #         save_path = 'models/model_params/{}-{}_{}_epoch{}_[hs={},topk={}].model'.format(hparams['name'],hparams['select'],hparams['scale'],hparams['epochs'], str(hparams['his_size']), str(hparams['k']))
    #     else:
    #         save_path = 'models/model_params/{}_{}_epoch{}_[hs={},topk={}].model'.format(hparams['name'],hparams['scale'],hparams['epochs'], str(hparams['his_size']), str(hparams['k']))
    #     torch.save(model.state_dict(), save_path)
    #     print("save success!")

    print("testing...")
    evaluate(model, hparams, loader_dev)

    if loader_validate is not None:
        print("validating...")
        evaluate(model, hparams, loader_validate)
    
    return model

def test(model, hparams):
    from utils.MIND import MIND_test
    tmp1 = hparams['batch_size']
    hparams['batch_size'] = 1
    tmp2 = model.batch_size
    model.batch_size = 1
    model.cdd_size = 1
    model.eval()

    dataset_test = MIND_test(hparams, '/home/peitian_zhang/Data/MIND/MINDlarge_test/news.tsv', '/home/peitian_zhang/Data/MIND/MINDlarge_test/behaviors.tsv')
    loader_test = DataLoader(dataset_test,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)
    tqdm_ = tqdm(enumerate(loader_test))
    with open('data/results/prediction.txt', 'w') as f:
        result = defaultdict(list)
        for step,x in tqdm_:
            pred = model(x)
            result[x['impression_index'][0].item()].append(pred.item())
        
        for k,v in result.items():
            rank_list = ss.rankdata(v, method='ordinal')
            line = str(k) + ' ' + str(rank_list) + '\n'
            f.write(line)
    
    hparam_list = ['name','scale','epochs','save_step','train_embedding','select','integrate','his_size','k','query_dim','value_dim','head_num']
    param_list = ['query_words','query_levels']
    with open('performance.log','a+') as f:
        # model_name = '{}-{}_{}_epoch{}_step{}_[hs={},topk={}]:'.format(hparams['name'],hparams['select'],hparams['scale'], str(hparams['epochs']), str(hparams['save_step']), str(hparams['his_size']), str(hparams['k']))
        # f.write(model_name + '\n')
        d = {}
        for k,v in hparams.items():
            if k in hparam_list:
                d[k] = v
        for name, param in model.named_parameters():
            if name in param_list:
                d[name] = tuple(param.shape)

        f.write(str(d)+'\n')
        f.write('\n')
        f.write('\n')   
    
    hparam['batch_size'] = tmp1
    model.batch_size = tmp2

    return res
    

def load_hparams(hparams):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--scale", dest="scale", help="data scale", choices=['demo','small','large'],required=True)
    parser.add_argument("-m","--mode", dest="mode", help="train or test", choices=['train','test','submit'], default='train')
    parser.add_argument("-e","--epochs", dest="epochs", help="epochs to train the model", type=int, default=10)

    parser.add_argument("-bs","--batch_size", dest="batch_size", help="batch size", type=int, default=100)
    parser.add_argument("-ts","--title_size", dest="title_size", help="news title size", type=int, default=20)
    parser.add_argument("-hs","--his_size", dest="his_size", help="history size", type=int, default=50)

    parser.add_argument("-c","--cuda", dest="cuda", help="device to run on", choices=['0','1'], default='0')
    parser.add_argument("-se","--save_each_epoch", dest="save_each_epoch", help="if clarified, save model of each epoch", action='store_true', default=True)
    parser.add_argument("-ss","--save_step", dest="save_step", help="if clarified, save model at the interval of given steps", type=int)
    parser.add_argument("-te","--train_embedding", dest="train_embedding", help="if clarified, word embedding will be fine-tuned", action='store_true', default=True)
    
    parser.add_argument("-k","--topk", dest="k", help="intend for topk baseline, if clarified, top k history are involved in interaction calculation", type=int, default=-1)
    parser.add_argument("-np","--npratio", dest="npratio", help="the number of unclicked news to sample when training", type=int, default=4)
    parser.add_argument("-mc","--metrics", dest="metrics", help="metrics for evaluating the model, if multiple metrics are needed, seperate with ','", type=str, default="group_auc,mean_mrr,ndcg@5,ndcg@10")

    parser.add_argument("--select", dest="select", help="choose model for selecting", choices=['pipeline','unified','gating'], default=None)
    parser.add_argument("--integrate", dest="integration", help="the way history filter is combined", choices=['gate','harmony'], default=None)

    parser.add_argument("-hn","--head_num", dest="head_num", help="number of multi-heads", type=int)
    parser.add_argument("-vd","--value_dim", dest="value_dim", help="dimension of projected value", type=int)
    parser.add_argument("-qd","--query_dim", dest="query_dim", help="dimension of projected query", type=int)

    parser.add_argument("-v","--validate", dest="validate", help="whether to validate training result on training dataset", action='store_true')

    # parser.add_argument("-dp","--dropout", dest="dropout", help="drop out probability", type=float, default=0.2)
    # parser.add_argument("-ed","--embedding_dim", dest="embedding_dim", help="dimension of word embedding", type=int, default=300)
    # parser.add_argument("-qd","--query_dim", dest="query_dim", help="dimension of query tensor", type=int, default=200)
    # parser.add_argument("-fn","--filter_num", dest="filter_num", help="the number of filters (out channels) for convolution", type=int, default=150)    


    args = parser.parse_args()

    hparams['scale'] = args.scale
    hparams['mode'] = args.mode
    hparams['device'] = 'cuda:' + args.cuda
    hparams['epochs'] = args.epochs
    hparams['batch_size'] = args.batch_size
    hparams['title_size'] = args.title_size
    hparams['his_size'] = args.his_size
    hparams['npratio'] = args.npratio
    hparams['metrics'] = args.metrics

    hparams['k'] = args.k
    hparams['select'] = args.select

    # intend for testing mode
    if args.select:            
        if args.save_step:
            hparams['save_path'] = 'models/model_params/{}-{}_{}_epoch{}_step{}_[hs={},topk={}].model'.format(hparams['name'],hparams['select'],hparams['scale'],hparams['epochs'],args.save_step, str(hparams['his_size']), str(hparams['k']))
        else:
            hparams['save_path'] = 'models/model_params/{}-{}_{}_epoch{}_[hs={},topk={}].model'.format(hparams['name'],hparams['select'],hparams['scale'],hparams['epochs'], str(hparams['his_size']), str(hparams['k']))
    else:
        if args.save_step:
            hparams['save_path'] = 'models/model_params/{}_{}_epoch{}_step{}_[hs={},topk={}].model'.format(hparams['name'],hparams['scale'],hparams['epochs'],args.save_step, str(hparams['his_size']), str(hparams['k']))
        else:
            hparams['save_path'] = 'models/model_params/{}_{}_epoch{}_[hs={},topk={}].model'.format(hparams['name'],hparams['scale'],hparams['epochs'], str(hparams['his_size']), str(hparams['k']))

    if args.head_num:
        hparams['head_num'] = args.head_num
    if args.value_dim:
        hparams['value_dim'] = args.value_dim
    if args.query_dim:
        hparams['query_dim'] = args.query_dim
    if args.integration:
        hparams['integration'] = args.integration
    
    hparams['validate'] = args.validate

    hparams['save_step'] = args.save_step
    hparams['save_each_epoch'] = args.save_each_epoch
    hparams['train_embedding'] = args.train_embedding

    return hparams

def prepare(hparams, validate=False, path='/home/peitian_zhang/Data/MIND'):
    from .MIND import MIND_iter, MIND_map
    """ prepare dataloader and several paths
    
    Args:
        hparams(dict): hyper parameters
    
    Returns:
        loader_train
        loader_dev
        loader_validate
    """
    news_file_train = path+'/MIND'+hparams['scale']+'_train/news.tsv'
    news_file_test = path+'/MIND'+hparams['scale']+'_dev/news.tsv'
    news_file_pair = (news_file_train,news_file_test)

    behavior_file_train = path+'/MIND'+hparams['scale']+'_train/behaviors.tsv'
    behavior_file_test = path+'/MIND'+hparams['scale']+'_dev/behaviors.tsv'
    behavior_file_pair = (behavior_file_train,behavior_file_test)

    if not os.path.exists('data/dictionaries/vocab_{}_{}.pkl'.format(hparams['scale'],'_'.join(hparams['attrs']))):
        constructBasicDict(news_file_pair,behavior_file_pair,hparams['scale'],hparams['attrs'])

    dataset_train = MIND_map(hparams=hparams,news_file=news_file_train,behaviors_file=behavior_file_train)
    dataset_dev = MIND_iter(hparams=hparams,news_file=news_file_test,behaviors_file=behavior_file_test, mode='test')
    dataset_validate = MIND_iter(hparams=hparams,news_file=news_file_train,behaviors_file=behavior_file_train, mode='train')

    vocab = dataset_train.vocab
    embedding = GloVe(dim=300,cache='.vector_cache')
    vocab.load_vectors(embedding)

    loader_train = DataLoader(dataset_train,batch_size=hparams['batch_size'],shuffle=True,pin_memory=True,num_workers=8,drop_last=True)
    loader_dev = DataLoader(dataset_dev,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)
    
    if validate:
        loader_validate = DataLoader(dataset_validate,batch_size=hparams['batch_size'],pin_memory=True,num_workers=0,drop_last=True)
        return vocab, loader_train, loader_dev, loader_validate
    else:
        return vocab, loader_train, loader_dev

def analyse(hparams, path='/home/peitian_zhang/Data/MIND'):
    """
        analyse over MIND
    """
    avg_title_length = 0
    avg_abstract_length = 0
    avg_his_length = 0
    avg_imp_length = 0
    cnt_his_lg_50 = 0
    cnt_his_eq_0 = 0
    cnt_imp_multi = 0

    news_file = path+'/MIND{}_{}/news.tsv'.format(hparams['scale'],hparams['mode'])

    behavior_file = path+'/MIND{}_{}/behaviors.tsv'.format(hparams['scale'],hparams['mode'])

    with open(news_file,"r",encoding='utf-8') as rd:
        count = 0
        for idx in rd:
            nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split('\t')
            avg_title_length += len(title.split(' '))
            avg_abstract_length += len(ab.split(' '))
            count += 1
    avg_title_length = avg_title_length/count
    avg_abstract_length = avg_abstract_length/count

    with open(behavior_file, "r", encoding='utf-8') as rd:
        count = 0
        for idx in rd:
            uid, time, history, impr = idx.strip("\n").split('\t')[-4:]
            his = history.split(' ')
            imp = impr.split(' ')
            if len(his) > 50:
                cnt_his_lg_50 += 1
            if len(imp) > 50:
                cnt_imp_multi += 1
            if not his[0]:
                cnt_his_eq_0 += 1
            avg_his_length += len(his)
            avg_imp_length += len(imp)
            count += 1
    avg_his_length = avg_his_length/count
    avg_imp_length = avg_imp_length/count

    print("avg_title_length:{}\n avg_abstract_length:{}\n avg_his_length:{}\n avg_impr_length:{}\n cnt_his_lg_50:{}\n cnt_his_eq_0:{}\n cnt_imp_multi:{}".format(avg_title_length,avg_abstract_length,avg_his_length,avg_imp_length,cnt_his_lg_50,cnt_his_eq_0,cnt_imp_multi))