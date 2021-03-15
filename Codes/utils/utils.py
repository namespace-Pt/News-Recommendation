import random
import re
import os
import math
import json
import pickle
import torch
import argparse
import logging
import pandas as pd
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import scipy.stats as ss
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, get_worker_info

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")


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


def word_tokenize_vocab(sent, vocab):
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


def news_token_generator(news_file_list, tokenizer, attrs):
    ''' merge and deduplicate training news and testing news then iterate, collect attrs into a single sentence and generate it

    Args: 
        tokenizer: torchtext.data.utils.tokenizer
        attrs: list of attrs to be collected and yielded
    Returns: 
        a generator over attrs in news
    '''
    news_df_list = []
    for f in news_file_list:
        news_df_list.append(pd.read_table(f, index_col=None, names=[
                            'newsID', 'category', 'subcategory', 'title', 'abstract', 'url', 'entity_title', 'entity_abstract'], quoting=3))

    news_df = pd.concat(news_df_list).drop_duplicates()
    news_iterator = news_df.iterrows()

    for _, i in news_iterator:
        content = []
        for attr in attrs:
            content.append(i[attr])

        yield tokenizer(' '.join(content))


def constructVocab(news_file_list, attrs):
    """
        Build field using torchtext for tokenization

    Returns:
        torchtext.vocabulary
    """
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        news_token_generator(news_file_list, tokenizer, attrs))

    output = open(
        'data/dictionaries/vocab_{}.pkl'.format(','.join(attrs)), 'wb')
    pickle.dump(vocab, output)
    output.close()


def constructNid2idx(news_file, scale, mode):
    """
        Construct news to newsID dictionary, index starting from 1
    """
    nid2index = {}

    news_df = pd.read_table(news_file, index_col=None, names=[
                            'newsID', 'category', 'subcategory', 'title', 'abstract', 'url', 'entity_title', 'entity_abstract'], quoting=3)

    for v in news_df['newsID']:
        if v in nid2index:
            continue
        nid2index[v] = len(nid2index) + 1

    h = open('data/dictionaries/nid2idx_{}_{}.json'.format(scale, mode), 'w')
    json.dump(nid2index, h, ensure_ascii=False)
    h.close()


def constructUid2idx(behavior_file_list, scale):
    """
        Construct user to userID dictionary, index starting from 1
    """
    uid2index = {}

    user_df_list = []
    for f in behavior_file_list:
        user_df_list.append(pd.read_table(f, index_col=None, names=[
                            'imprID', 'uid', 'time', 'hisstory', 'abstract', 'impression'], quoting=3))

    user_df = pd.concat(user_df_list).drop_duplicates()

    for v in user_df['uid']:
        if v in uid2index:
            continue
        uid2index[v] = len(uid2index) + 1

    h = open('data/dictionaries/uid2idx_{}.json'.format(scale), 'w')
    json.dump(uid2index, h, ensure_ascii=False)
    h.close()


def constructBasicDict(attrs=['title'], path='/home/peitian_zhang/Data/MIND'):
    """
        construct basic dictionary
    """
    news_file_list2 = [path + '/MINDlarge_train/news.tsv', path +
                       '/MINDlarge_dev/news.tsv', path + '/MINDlarge_test/news.tsv']
    constructVocab(news_file_list2, attrs)

    for scale in ['demo', 'small', 'large']:
        news_file_list = [path + '/MIND{}_train/news.tsv'.format(
            scale), path + '/MIND{}_dev/news.tsv'.format(scale), path + '/MIND{}_test/news.tsv'.format(scale)]
        behavior_file_list = [path + '/MIND{}_train/behaviors.tsv'.format(
            scale), path + '/MIND{}_dev/behaviors.tsv'.format(scale), path + '/MIND{}_test/behaviors.tsv'.format(scale)]

        if scale == 'large':
            news_file_train = news_file_list[0]
            news_file_dev = news_file_list[1]
            news_file_test = news_file_list[2]

            constructNid2idx(news_file_train, scale, 'train')
            constructNid2idx(news_file_dev, scale, 'dev')
            constructNid2idx(news_file_test, scale, 'test')

            constructUid2idx(behavior_file_list, scale)

        else:
            news_file_list = news_file_list[0:2]

            news_file_train = news_file_list[0]
            news_file_dev = news_file_list[1]

            constructNid2idx(news_file_train, scale, 'train')
            constructNid2idx(news_file_dev, scale, 'dev')

            behavior_file_list = behavior_file_list[0:2]
            constructUid2idx(behavior_file_list, scale)


def tailorData(tsvFile, num):
    ''' tailor num rows of tsvFile to create demo data file

    Args: 
        tsvFile: str of data path
    Returns: 
        create tailored data file
    '''
    pattern = re.search('(.*)MIND(.*)_(.*)/(.*).tsv', tsvFile)

    directory = pattern.group(1)
    mode = pattern.group(3)
    behavior_file = pattern.group(4)

    behavior_file = directory + 'MINDdemo' + \
        '_{}/'.format(mode) + behavior_file + '.tsv'

    f = open(behavior_file, 'w', encoding='utf-8')
    count = 0
    with open(tsvFile, 'r', encoding='utf-8') as g:
        for line in g:
            if count >= num:
                f.close()
                break
            f.write(line)
            count += 1
    news_file = re.sub('behaviors', 'news', tsvFile)
    news_file_new = re.sub('behaviors', 'news', behavior_file)

    os.system("cp {} {}".format(news_file, news_file_new))
    logging.info("tailored {} behaviors to {}, copied news file also".format(num, behavior_file))
    return

def getId2idx(file):
    """
        get Id2idx dictionary from json file 
    """
    g = open(file, 'r', encoding='utf-8')
    dic = json.load(g)
    g.close()
    return dic


def getVocab(file):
    """
        get Vocabulary from pkl file
    """
    g = open(file, 'rb')
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


def getLabel(model, x):
    """
        parse labels to label indexes, used in NLLoss
    """
    if model.cdd_size > 1:
        index = torch.arange(0, model.cdd_size, device=model.device).expand(
            model.batch_size, -1)
        label = x['labels'] == 1
        label = index[label]
    else:
        label = x['labels']

    return label


def my_collate(data):
    excluded = ['impression_index']
    result = defaultdict(list)
    for d in data:
        for k, v in d.items():
            result[k].append(v)
    for k, v in result.items():
        if k not in excluded:
            result[k] = torch.tensor(v)
        else:
            continue
    return dict(result)


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_impr_indexes = dataset.impr_indexes

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil(len(overall_impr_indexes) /
                               float(worker_info.num_workers)))
    worker_id = worker_info.id
    start = worker_id * per_worker
    end = (worker_id + 1) * per_worker

    dataset.impr_indexes = dataset.impr_indexes[start: end]


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


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
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
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def run_eval(model, dataloader, interval):
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

    for i, batch_data_input in tqdm(enumerate(dataloader)):
        pred = model.forward(batch_data_input).squeeze(dim=-1).tolist()
        preds.extend(pred)
        label = batch_data_input['labels'].squeeze(dim=-1).tolist()
        labels.extend(label)
        imp_indexes.extend(batch_data_input['impression_index'])

    all_keys = list(set(imp_indexes))
    all_keys.sort()
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, imp_indexes):
        group_labels[k].append(l)
        group_preds[k].append(p)

    all_labels = []
    all_preds = []

    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return group_labels.keys(), all_labels, all_preds


@torch.no_grad()
def _eval_mtp(i, model, hparams, dataloader, result_list):
    """evaluate in multi-processing

    Args:
        i(int) Subprocess No.
        model(nn.Module)
        hparams(dict)
        dataloader(torch.utils.data.DataLoader)
        result_list(torch.multiprocessing.Manager().list()): contain evaluation results of every subprocesses
    """

    logging.info(
        "[No.{}, PID:{}] loading model parameters...".format(i, os.getpid()))

    step = hparams['save_step'][i]
    save_path = 'models/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model'.format(
        hparams['name'], hparams['scale'], hparams['epochs'], step, str(hparams['his_size']), str(hparams['k']))
    model.load_state_dict(torch.load(
        save_path, map_location=hparams['device']))

    logging.info("[No.{}, PID:{}] evaluating...".format(i, os.getpid()))

    imp_indexes, labels, preds = run_eval(model, dataloader, 10)
    res = cal_metric(labels, preds, hparams['metrics'].split(','))

    res['step'] = step
    logging.info("\nevaluation results of process NO.{} is {}".format(i, res))

    result_list.append(res)


@torch.no_grad()
def evaluate(model, hparams, dataloader, load=False, interval=100):
    """Evaluate the given file and returns some evaluation metrics.

    Args:
        model(nn.Module)
        hparams(dict)
        dataloader(torch.utils.data.DataLoader): provide data
        load(bool): whether to load model in hparams['save_path']
        interval(int): within each epoch, the interval of steps to display loss

    Returns:
        dict: A dictionary contains evaluation metrics.
    """
    hparam_list = ['name', 'scale', 'epochs', 'train_embedding', 'select',
                   'integrate', 'his_size', 'k', 'query_dim', 'value_dim', 'head_num']
    param_list = ['query_words', 'query_levels']

    model.eval()
    cdd_size = model.cdd_size
    model.cdd_size = 1

    steps = len(hparams['save_step'])

    if steps == 1:
        if load:
            logging.info("loading model...")
            save_path = 'models/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model'.format(
                hparams['name'], hparams['scale'], hparams['epochs'], hparams['save_step'][0], hparams['his_size'], hparams['k'])
            state_dict = torch.load(save_path, map_location=hparams['device'])
            state_dict = {k: v for k, v in state_dict.items() if k not in [
                'news_reprs.weight', 'news_embeddings.weight']}
            model.load_state_dict(state_dict, strict=False)

        logging.info("evaluating...")

        imp_indexes, labels, preds = run_eval(model, dataloader, interval)
        res = cal_metric(labels, preds, hparams['metrics'].split(','))

        res['step'] = hparams['save_step'][0]

        logging.info("evaluation results:{}".format(res))
        with open('performance.log', 'a+') as f:
            d = {}
            for k, v in hparams.items():
                if k in hparam_list:
                    d[k] = v
            for name, param in model.named_parameters():
                if name in param_list:
                    d[name] = tuple(param.shape)

            f.write(str(d)+'\n')
            f.write(str(res) + '\n')
            f.write('\n')

        model.train()
        model.cdd_size = cdd_size
        return res

    elif steps > 1:
        logging.info("evaluating in {} processes...".format(steps))
        model.share_memory()
        res_list = mp.Manager().list()
        mp.spawn(_eval_mtp, args=(model, hparams,
                                  dataloader, res_list), nprocs=steps)
        with open('performance.log', 'a+') as f:
            d = {}
            for k, v in hparams.items():
                if k in hparam_list:
                    d[k] = v
            for name, param in model.named_parameters():
                if name in param_list:
                    d[name] = tuple(param.shape)
            f.write(str(d)+'\n')

            for result in res_list:
                f.write(str(result) + '\n')
            f.write('\n')

    model.cdd_size = cdd_size


def run_train(model, dataloader, optimizer, loss_func, hparams, writer=None, interval=100, save_step=None):
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
    total_steps = 0

    for epoch in range(hparams['epochs']):
        epoch_loss = 0
        tqdm_ = tqdm(enumerate(dataloader))
        for step, x in tqdm_:
            pred = model(x)
            label = getLabel(model, x)
            loss = loss_func(pred, label)

            epoch_loss += loss
            total_loss += loss

            loss.backward()
            optimizer.step()

            if step % interval == 0:

                tqdm_.set_description(
                    "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                if writer:
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param, step)

                    writer.add_scalar('data_loss',
                                      total_loss/total_steps)
            optimizer.zero_grad()

            if save_step:
                if step % save_step == 0 and step > 0:
                    save_path = 'models/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model'.format(
                        hparams['name'], hparams['scale'], epoch + 1, step, hparams['his_size'], hparams['k'])
                    if not os.path.exists(save_path):
                        os.makedirs('/'.join(save_path.split('/')[:-1]))

                    torch.save(model.state_dict(), save_path)
                    logging.info(
                        "saved model of step {} at epoch {}".format(step, epoch+1))

            total_steps += 1

        if writer:
            writer.add_scalar('epoch_loss', epoch_loss, epoch)

        save_path = 'models/model_params/{}/{}_epoch{}_step0_[hs={},topk={}].model'.format(
            hparams['name'], hparams['scale'], epoch+1, hparams['his_size'], hparams['k'])

        state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k not in [
            'news_reprs.weight', 'news_embeddings.weight']}
        torch.save(state_dict, save_path)
        logging.info("saved model of epoch {}".format(epoch+1))

    return model


def train(model, hparams, loaders, tb=False, interval=100):
    """ wrap training process

    Args:
        model(torch.nn.Module): the model to be trained
        loaders(list): list of torch.utils.data.DataLoader
        hparams(dict): hyper paramaters
        en: shell parameter
    """
    model.train()
    writer = None

    if tb:
        if hparams['select']:
            writer = SummaryWriter('data/tb/{}-{}/{}/{}/'.format(
                hparams['name'], hparams['select'], hparams['scale'], datetime.now().strftime("%Y%m%d-%H")))
        else:
            writer = SummaryWriter('data/tb/{}/{}/{}/'.format(
                hparams['name'], hparams['scale'], datetime.now().strftime("%Y%m%d-%H")))

    logging.info("training...")
    loss_func = getLoss(model)
    if 'learning_rate' in hparams:
        learning_rate = hparams['learning_rate']
    else:
        learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = run_train(model, loaders[0], optimizer, loss_func, hparams,
                      writer=writer, interval=interval, save_step=hparams['save_step'][0])

    evaluate(model, hparams, loaders[1], load=False)

    # loader_train, loader_dev, loader_validate
    if len(loaders) > 2:
        logging.info("validating...")
        evaluate(model, hparams, loaders[2])

    return model


@torch.no_grad()
def test(model, hparams, loader_test):
    """ test the model on test dataset of MINDlarge

    Args:
        model
        hparams
        loader_test: DataLoader of MINDlarge_test
    """
    save_path = 'models/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model'.format(
        hparams['name'], hparams['scale'], hparams['epochs'], hparams['save_step'][0], hparams['his_size'], hparams['k'])
    state_dict = torch.load(save_path, map_location=hparams['device'])
    state_dict = {k: v for k, v in state_dict.items() if k not in [
        'news_reprs.weight', 'news_embeddings.weight']}
    model.load_state_dict(state_dict, strict=False)

    logging.info("testing...")
    model.cdd_size = 1
    model.eval()

    save_path = 'data/results/prediction={}_{}_epoch{}_step{}_[hs={},topk={}].txt'.format(
        hparams['name'], hparams['scale'], hparams['epochs'], hparams['save_step'][0], hparams['his_size'], hparams['k'])
    with open(save_path, 'w') as f:
        preds = []
        imp_indexes = []
        for i, x in tqdm(enumerate(loader_test)):
            preds.extend(model.forward(x).tolist())
            imp_indexes.extend(x['impression_index'])

        all_keys = list(set(imp_indexes))
        all_keys.sort()
        group_preds = {k: [] for k in all_keys}

        for i, p in zip(imp_indexes, preds):
            group_preds[i].append(p)

        for k, v in group_preds.items():
            array = np.asarray(v)
            rank_list = ss.rankdata(1 - array, method='ordinal')
            line = str(k) + ' [' + ','.join([str(i)
                                             for i in rank_list]) + ']' + '\n'
            f.write(line)

    logging.info("written to prediction!")

    hparam_list = ['name', 'scale', 'epochs', 'save_step', 'train_embedding',
                   'select', 'integrate', 'his_size', 'k', 'query_dim', 'value_dim', 'head_num']
    param_list = ['query_words', 'query_levels']
    with open('performance.log', 'a+') as f:
        d = {}
        for k, v in hparams.items():
            if k in hparam_list:
                d[k] = v
        for name, param in model.named_parameters():
            if name in param_list:
                d[name] = tuple(param.shape)

        f.write(str(d)+'\n')
        f.write('\n')
        f.write('\n')


def tune(model, hparams, loaders, best_auc=0):
    """ tune hyper parameters

    Args:
        step_list(list): the step of training model
    """
    logging.info("current hyper parameter settings are:{}".format(hparams))

    loader_train = loaders[0]
    loader_dev = loaders[1]

    loss_func = getLoss(model)

    if 'learning_rate' in hparams:
        learning_rate = hparams['learning_rate']
    else:
        learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(hparams['epochs']):
        epoch_loss = 0
        tqdm_ = tqdm(enumerate(loader_train))
        for step, x in tqdm_:
            pred = model(x)
            label = getLabel(model, x)
            loss = loss_func(pred, label)

            tqdm_.set_description("epoch {:d} , step {:d} , loss: {:.4f}".format(
                epoch+1, step, epoch_loss / (step+1)))

            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step > 19999 and step % 2000 == 0:
                save_path = 'models/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model'.format(
                    hparams['name'], hparams['scale'], epoch + 1, step, str(hparams['his_size']), str(hparams['k']))
                torch.save(model.state_dict(), save_path)
                logging.info(
                    "saved model of step {} at epoch {}".format(step, epoch+1))

        # without evaluating, only training

        # logging.info("evaluating in {} processes...".format(len(hparams['save_step'])))
        # with torch.no_grad():
        #     model.share_memory()
        #     res_list = mp.Manager().list()
            # mp.spawn(_eval_mtp, args=(model, hparams, loader_dev, res_list), nprocs=len(hparams['step_list']))

        #     with open('sfi-performance.log','a+') as f:
        #         for result in res_list:
        #             if result['auc'] > best_auc:
        #                 best_auc = result['auc']

        #                 d = {}
        #                 for k,v in hparams.items():
        #                     if k in hparam_list:
        #                         d[k] = v

        #                 for name, param in model.named_parameters():
        #                     if name in param_list:
        #                         d[name] = tuple(param.shape)

        #                 f.write(str(d)+'\n')
        #                 f.write(str(result) +'\n')
        #                 f.write('\n')

    return best_auc


def load_hparams(hparams):
    """ 
        customize hyper parameters in command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scale", dest="scale", help="data scale",
                        choices=['demo', 'small', 'large'], required=True)
    parser.add_argument("-m", "--mode", dest="mode", help="train or test",
                        choices=['train', 'dev', 'test', 'tune'], default='train')
    parser.add_argument("-e", "--epochs", dest="epochs",
                        help="epochs to train the model", type=int, default=10)

    parser.add_argument("-bs", "--batch_size", dest="batch_size",
                        help="batch size", type=int, default=100)
    parser.add_argument("-ts", "--title_size", dest="title_size",
                        help="news title size", type=int, default=20)
    parser.add_argument("-hs", "--his_size", dest="his_size",
                        help="history size", type=int, default=50)

    parser.add_argument("--cuda", dest="cuda",
                        help="device to run on", choices=['0', '1'], default='0')
    # parser.add_argument("--save_each_epoch", dest="save_each_epoch", help="if clarified, save model of each epoch", default=True)
    parser.add_argument("--save_step", dest="save_step",
                        help="if clarified, save model at the interval of given steps", type=str, default='0')
    parser.add_argument("--train_embedding", dest="train_embedding",
                        help="if clarified, word embedding will be fine-tuned", default=True)
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                        help="learning rate when training", type=float, default=1e-3)
    # parser.add_argument("--nni", dest="use_nni", help="if clarified, the nni package will be used", action='store_true')

    parser.add_argument("-np", "--npratio", dest="npratio",
                        help="the number of unclicked news to sample when training", type=int, default=4)
    parser.add_argument("-mc", "--metrics", dest="metrics",
                        help="metrics for evaluating the model, if multiple metrics are needed, seperate with ','", type=str, default="auc,mean_mrr,ndcg@5,ndcg@10")

    # parser.add_argument("--level", dest="level", help="intend for fim baseline, if clarified, level representations will be learnt for a token", type=int)
    parser.add_argument(
        "--topk", dest="k", help="intend for topk baseline, if clarified, top k history are involved in interaction calculation", type=int, default=0)
    parser.add_argument("--select", dest="select", help="choose model for selecting",
                        choices=['pipeline1', 'pipeline2', 'unified', 'gating'], default=None)
    parser.add_argument("--integrate", dest="integration",
                        help="the way history filter is combined", choices=['gate', 'harmony'], default=None)
    parser.add_argument("--encoder", dest="encoder", help="choose encoder", choices=['fim', 'npa', 'mha'], default=None)

    parser.add_argument("-hn", "--head_num", dest="head_num",
                        help="number of multi-heads", type=int)
    parser.add_argument("-vd", "--value_dim", dest="value_dim",
                        help="dimension of projected value", type=int)
    parser.add_argument("-qd", "--query_dim", dest="query_dim",
                        help="dimension of projected query", type=int)

    parser.add_argument("--attrs", dest="attrs",
                        help="clarified attributes of news will be yielded by dataloader, seperate with comma", type=str, default='title')
    parser.add_argument("-v", "--validate", dest="validate",
                        help="if clarified, evaluate the model on training set", action='store_true')
    parser.add_argument("-nid", "--news_id", dest="news_id",
                        help="if clarified, the id of news will be yielded by dataloader", action='store_true')

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
    hparams['npratio'] = args.npratio
    hparams['metrics'] = args.metrics
    hparams['learning_rate'] = args.learning_rate

    hparams['his_size'] = args.his_size
    hparams['k'] = args.k
    hparams['select'] = args.select

    hparams['save_step'] = [int(i) for i in args.save_step.split(',')]

    if args.head_num:
        hparams['head_num'] = args.head_num
    if args.value_dim:
        hparams['value_dim'] = args.value_dim
    if args.query_dim:
        hparams['query_dim'] = args.query_dim
    if args.integration:
        hparams['integration'] = args.integration
    if args.encoder:
        hparams['encoder'] = args.encoder

    # if args.level:
    #     hparams['level'] = args.level

    hparams['attrs'] = args.attrs.split(',')
    hparams['validate'] = args.validate
    hparams['news_id'] = args.news_id

    # hparams['save_each_epoch'] = args.save_each_epoch
    hparams['train_embedding'] = args.train_embedding

    return hparams


def generate_hparams(hparams, config):
    """ update hyper parameters with values in config

    Args:
        hparams
        config(dict)

    Returns:
        hparams
    """
    val_pool = []
    for vals in config.values():
        val_pool.append(vals)
    key_list = list(config.keys())

    for vals in product(*val_pool):
        for i, val in enumerate(vals):
            hparams[key_list[i]] = val

        yield hparams


def prepare(hparams, path='/home/peitian_zhang/Data/MIND', shuffle=True, news=False):
    from .MIND import MIND, MIND_test, MIND_news
    """ prepare dataloader and several paths
    
    Args:
        hparams(dict): hyper parameters
    
    Returns:
        vocab
        loaders(list of dataloaders): 0-loader_train/test/dev, 1-loader_dev, 2-loader_validate
    """
    logging.info("Hyper Parameters are\n{}".format(hparams))

    logging.info("preparing dataset...")

    if news:
        path = '/home/peitian_zhang/Data/MIND'
        news_file_train = path + \
            '/MIND{}_train/news.tsv'.format(hparams['scale'])
        news_file_dev = path+'/MIND{}_dev/news.tsv'.format(hparams['scale'])

        dataset_train = MIND_news(hparams, news_file_train)
        loader_news_train = DataLoader(
            dataset_train, batch_size=hparams['batch_size'], pin_memory=True, num_workers=8, drop_last=False, collate_fn=my_collate)

        dataset_dev = MIND_news(hparams, news_file_dev)
        loader_news_dev = DataLoader(
            dataset_dev, batch_size=hparams['batch_size'], pin_memory=True, num_workers=8, drop_last=False, collate_fn=my_collate)

        vocab = dataset_train.vocab
        embedding = GloVe(dim=300, cache='.vector_cache')
        vocab.load_vectors(embedding)

        if hparams['scale'] == 'large':
            news_file_test = path + \
                '/MIND{}_test/news.tsv'.format(hparams['scale'])
            dataset_test = MIND_news(hparams, news_file_test)
            loader_news_test = DataLoader(
                dataset_test, batch_size=hparams['batch_size'], pin_memory=True, num_workers=8, drop_last=False, collate_fn=my_collate)

            return vocab, [loader_news_train, loader_news_dev, loader_news_test]

        return vocab, [loader_news_train, loader_news_dev]

    if hparams['mode'] in ['train', 'tune']:
        news_file_train = path+'/MIND'+hparams['scale']+'_train/news.tsv'
        news_file_dev = path+'/MIND'+hparams['scale']+'_dev/news.tsv'

        behavior_file_train = path+'/MIND' + \
            hparams['scale']+'_train/behaviors.tsv'
        behavior_file_dev = path+'/MIND'+hparams['scale']+'_dev/behaviors.tsv'

        dataset_train = MIND(hparams=hparams, news_file=news_file_train,
                             behaviors_file=behavior_file_train, shuffle=shuffle)
        dataset_dev = MIND(hparams=hparams, news_file=news_file_dev,
                           behaviors_file=behavior_file_dev, npratio=0)

        loader_train = DataLoader(dataset_train, batch_size=hparams['batch_size'], pin_memory=True,
                                  num_workers=8, drop_last=False, collate_fn=my_collate, worker_init_fn=worker_init_fn)
        loader_dev = DataLoader(dataset_dev, batch_size=hparams['batch_size'], pin_memory=True,
                                num_workers=8, drop_last=False, collate_fn=my_collate, worker_init_fn=worker_init_fn)

        vocab = dataset_train.vocab
        embedding = GloVe(dim=300, cache='.vector_cache')
        vocab.load_vectors(embedding)

        if hparams['validate']:
            dataset_validate = MIND(
                hparams=hparams, news_file=news_file_train, behaviors_file=behavior_file_train, npratio=0)
            loader_validate = DataLoader(dataset_validate, batch_size=hparams['batch_size'], pin_memory=True,
                                         num_workers=8, drop_last=False, collate_fn=my_collate, worker_init_fn=worker_init_fn)
            return vocab, [loader_train, loader_dev, loader_validate]
        else:
            return vocab, [loader_train, loader_dev]

    elif hparams['mode'] == 'dev':
        news_file_dev = path+'/MIND'+hparams['scale']+'_dev/news.tsv'
        behavior_file_dev = path+'/MIND'+hparams['scale']+'_dev/behaviors.tsv'
        dataset_dev = MIND(hparams=hparams, news_file=news_file_dev,
                           behaviors_file=behavior_file_dev, npratio=0)
        loader_dev = DataLoader(dataset_dev, batch_size=hparams['batch_size'], pin_memory=True,
                                num_workers=8, drop_last=False, collate_fn=my_collate, worker_init_fn=worker_init_fn)
        vocab = dataset_dev.vocab
        embedding = GloVe(dim=300, cache='.vector_cache')
        vocab.load_vectors(embedding)

        return vocab, [loader_dev]

    elif hparams['mode'] == 'test':
        dataset_test = MIND_test(hparams, '/home/peitian_zhang/Data/MIND/MINDlarge_test/news.tsv',
                                 '/home/peitian_zhang/Data/MIND/MINDlarge_test/behaviors.tsv')
        loader_test = DataLoader(dataset_test, batch_size=hparams['batch_size'], pin_memory=True,
                                 num_workers=8, drop_last=False, collate_fn=my_collate, worker_init_fn=worker_init_fn)

        vocab = dataset_test.vocab
        embedding = GloVe(dim=300, cache='.vector_cache')
        vocab.load_vectors(embedding)

        return vocab, [loader_test]


def pipeline_encode(model, hparams, loaders):
    news_num_dict = {
        'demo': {
            'train': 51282,
            'dev': 42416
        },
        'small': {
            'train': 51282,
            'dev': 42416
        },
        'large': {
            'train': 101527,
            'dev': 72023,
            'test': 120961
        }
    }
    news_num_train = news_num_dict[hparams['scale']]['train']

    news_reprs = torch.zeros((news_num_train + 1, model.filter_num))
    news_embeddings = torch.zeros(
        (news_num_train + 1, model.signal_length, model.level, model.filter_num))

    for x in tqdm(loaders[0]):
        embedding, repr = model(x)
        for i in range(embedding.shape[0]):
            news_reprs[x['news_id'][i]] = repr[i]
            news_embeddings[x['news_id'][i]] = embedding[i]

    torch.save(news_reprs, 'data/tensors/news_reprs_{}_train-[{}].tensor'.format(
        hparams['scale'], hparams['name']))
    torch.save(news_embeddings, 'data/tensors/news_embeddings_{}_train-[{}].tensor'.format(
        hparams['scale'], hparams['name']))

    news_num_dev = news_num_dict[hparams['scale']]['dev']

    news_reprs = torch.zeros((news_num_dev + 1, model.filter_num))
    news_embeddings = torch.zeros(
        (news_num_dev + 1, model.signal_length, model.level, model.filter_num))

    for x in tqdm(loaders[1]):
        embedding, repr = model(x)
        for i in range(embedding.shape[0]):
            news_reprs[x['news_id'][i]] = repr[i]
            news_embeddings[x['news_id'][i]] = embedding[i]

    torch.save(news_reprs, 'data/tensors/news_reprs_{}_dev-[{}].tensor'.format(
        hparams['scale'], hparams['name']))
    torch.save(news_embeddings, 'data/tensors/news_embeddings_{}_dev-[{}].tensor'.format(
        hparams['scale'], hparams['name']))

    if hparams['scale'] == 'large':
        news_num_test = news_num_dict[hparams['scale']]['test']

        news_reprs = torch.zeros((news_num_test + 1, model.filter_num))
        news_embeddings = torch.zeros(
            (news_num_test + 1, model.signal_length, model.level, model.filter_num))

        for x in tqdm(loaders[2]):
            embedding, repr = model(x)
            for i in range(embedding.shape[0]):
                news_reprs[x['news_id'][i]] = repr[i]
                news_embeddings[x['news_id'][i]] = embedding[i]

        torch.save(news_reprs, 'data/tensors/news_reprs_{}_test-[{}].tensor'.format(
            hparams['scale'], hparams['name']))
        torch.save(news_embeddings, 'data/tensors/news_embeddings_{}_test-[{}].tensor'.format(
            hparams['scale'], hparams['name']))

    logging.info('successfully encoded news!')


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

    news_file = path + \
        '/MIND{}_{}/news.tsv'.format(hparams['scale'], hparams['mode'])

    behavior_file = path + \
        '/MIND{}_{}/behaviors.tsv'.format(hparams['scale'], hparams['mode'])

    with open(news_file, "r", encoding='utf-8') as rd:
        count = 0
        for idx in rd:
            nid, vert, subvert, title, ab, url, _, _ = idx.strip(
                "\n").split('\t')
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

    print("avg_title_length:{}\n avg_abstract_length:{}\n avg_his_length:{}\n avg_impr_length:{}\n cnt_his_lg_50:{}\n cnt_his_eq_0:{}\n cnt_imp_multi:{}".format(
        avg_title_length, avg_abstract_length, avg_his_length, avg_imp_length, cnt_his_lg_50, cnt_his_eq_0, cnt_imp_multi))
