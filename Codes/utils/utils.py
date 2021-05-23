import random
import re
import os
import sys
import json
import pickle
import torch
import argparse
import logging
import subprocess
import pandas as pd
import torch.nn as nn
# import torch.multiprocessing as mp
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
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")

hparam_list = ["name", "scale", "select", "integrate", "his_size", "k", "contra_num", "threshold", "checkpoint", "epochs", "save_step", "learning_rate"]
param_list = ["query_words", "query_levels", "CoAttention.weight", "selectionProject.weight"]

class TripletMarginLoss():
    """
        triplet margin loss based on dot product
    """
    def __init__(self, margin=1):
        self.margin = margin

    def __call__(self, anchor, pos, neg):
        """ calculate loss

        Args:
            anchor: the anchor for positive and negtive samples, tensor of [N,E]
            pos: positive samples, tensor of [N,K,E]
            neg: negative samples, tensor of [N,K,E]
        """

        anchor = anchor.unsqueeze(dim=1)
        pos_relevance = anchor.matmul(pos.transpose(-2,-1)).squeeze(dim=1)
        neg_relevance = anchor.matmul(neg.transpose(-2,-1)).squeeze(dim=1)

        max_relevance = torch.max(torch.cat([pos_relevance,neg_relevance],dim=0).view(-1))
        pos_dist = max_relevance - pos_relevance
        neg_dist = max_relevance - neg_relevance

        margin_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0)
        loss = torch.mean(margin_dist)
        return loss

def tokenize(sent, vocab):
    """ Split sentence into wordID list using regex and vocabulary
    Args:
        sent (str): Input sentence
        vocab : vocabulary

    Return:
        list: word list
    """
    pat = re.compile(r"[-\w_]+|[.,!?;|]")
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
    """ merge and deduplicate training news and testing news then iterate, collect attrs into a single sentence and generate it

    Args:
        tokenizer: torchtext.data.utils.tokenizer
        attrs: list of attrs to be collected and yielded
    Returns:
        a generator over attrs in news
    """
    news_df_list = []
    for f in news_file_list:
        news_df_list.append(pd.read_table(f, index_col=None, names=[
                            "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3))

    news_df = pd.concat(news_df_list).drop_duplicates().dropna()
    news_iterator = news_df.iterrows()

    for _, i in news_iterator:
        content = []
        for attr in attrs:
            content.append(i[attr])

        yield tokenizer(" ".join(content))


def constructVocab(news_file_list, attrs):
    """
        Build field using torchtext for tokenization

    Returns:
        torchtext.vocabulary
    """
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        news_token_generator(news_file_list, tokenizer, attrs))

    output = open(
        "data/dictionaries/vocab_{}.pkl".format(",".join(attrs)), "wb")
    pickle.dump(vocab, output)
    output.close()


def constructNid2idx(news_file, scale, mode):
    """
        Construct news to newsID dictionary, index starting from 1
    """
    nid2index = {}

    news_df = pd.read_table(news_file, index_col=None, names=[
                            "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3)

    for v in news_df["newsID"]:
        if v in nid2index:
            continue
        nid2index[v] = len(nid2index) + 1

    h = open("data/dictionaries/nid2idx_{}_{}.json".format(scale, mode), "w")
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
                            "imprID", "uid", "time", "hisstory", "abstract", "impression"], quoting=3))

    user_df = pd.concat(user_df_list).drop_duplicates()

    for v in user_df["uid"]:
        if v in uid2index:
            continue
        uid2index[v] = len(uid2index) + 1

    h = open("data/dictionaries/uid2idx_{}.json".format(scale), "w")
    json.dump(uid2index, h, ensure_ascii=False)
    h.close()


def constructBasicDict(attrs=["title"], path="/home/peitian_zhang/Data/MIND"):
    """
        construct basic dictionary
    """
    news_file_list = [path + "/MINDlarge_train/news.tsv", path +
                       "/MINDlarge_dev/news.tsv", path + "/MINDlarge_test/news.tsv"]
    constructVocab(news_file_list, attrs)

    for scale in ["demo", "small", "large"]:
        news_file_list = [path + "/MIND{}_train/news.tsv".format(
            scale), path + "/MIND{}_dev/news.tsv".format(scale), path + "/MIND{}_test/news.tsv".format(scale)]
        behavior_file_list = [path + "/MIND{}_train/behaviors.tsv".format(
            scale), path + "/MIND{}_dev/behaviors.tsv".format(scale), path + "/MIND{}_test/behaviors.tsv".format(scale)]

        if scale == "large":
            news_file_train = news_file_list[0]
            news_file_dev = news_file_list[1]
            news_file_test = news_file_list[2]

            constructNid2idx(news_file_train, scale, "train")
            constructNid2idx(news_file_dev, scale, "dev")
            constructNid2idx(news_file_test, scale, "test")

            constructUid2idx(behavior_file_list, scale)

        else:
            news_file_list = news_file_list[0:2]

            news_file_train = news_file_list[0]
            news_file_dev = news_file_list[1]

            constructNid2idx(news_file_train, scale, "train")
            constructNid2idx(news_file_dev, scale, "dev")

            behavior_file_list = behavior_file_list[0:2]
            constructUid2idx(behavior_file_list, scale)


def constructVertOnehot():
    import pandas as pd
    path = "/home/peitian_zhang/Data/MIND"
    news_file_list = [path + "/MINDlarge_train/news.tsv", path +
                        "/MINDlarge_dev/news.tsv", path + "/MINDlarge_test/news.tsv"]
    news_df_list = []
    for f in news_file_list:
        news_df_list.append(pd.read_table(f, index_col=None, names=["newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3))

    news_df = pd.concat(news_df_list).drop_duplicates()

    vert = news_df["category"].unique()
    subvert = news_df["subcategory"].unique()
    vocab = getVocab("data/dictionaries/vocab_whole.pkl")
    vert2idx = {
        vocab[v]:i for i,v in enumerate(vert)
    }
    subvert2idx = {
        vocab[v]:i for i,v in enumerate(subvert)
    }
    vert2onehot = {}
    for k,v in vert2idx.items():
        a = np.zeros((len(vert2idx)))
        index = np.asarray([v])
        a[index] = 1
        vert2onehot[int(k)] = a.tolist()
    vert2onehot[1] = [0]*len(next(iter(vert2onehot.values())))

    subvert2onehot = {}
    for k,v in subvert2idx.items():
        a = np.zeros((len(subvert2idx)))
        index = np.asarray([v])
        a[index] = 1
        subvert2onehot[int(k)] = a.tolist()
    subvert2onehot[1] = [0]*len(next(iter(subvert2onehot.values())))

    json.dump(vert2onehot, open("data/dictionaries/vert2onehot.json","w"),ensure_ascii=False)
    json.dump(subvert2onehot, open("data/dictionaries/subvert2onehot.json","w"),ensure_ascii=False)

def tailorData(tsvFile, num):
    """ tailor num rows of tsvFile to create demo data file

    Args:
        tsvFile: str of data path
    Returns:
        create tailored data file
    """
    pattern = re.search("(.*)MIND(.*)_(.*)/(.*).tsv", tsvFile)

    directory = pattern.group(1)
    mode = pattern.group(3)
    behavior_file = pattern.group(4)

    if not os.path.exists(directory + "MINDdemo" + "_{}/".format(mode)):
        os.mkdir(directory + "MINDdemo" + "_{}/".format(mode))

    behavior_file = directory + "MINDdemo" + \
        "_{}/".format(mode) + behavior_file + ".tsv"

    f = open(behavior_file, "w", encoding="utf-8")
    count = 0
    with open(tsvFile, "r", encoding="utf-8") as g:
        for line in g:
            if count >= num:
                f.close()
                break
            f.write(line)
            count += 1
    news_file = re.sub("behaviors", "news", tsvFile)
    news_file_new = re.sub("behaviors", "news", behavior_file)

    os.system("cp {} {}".format(news_file, news_file_new))
    logging.info("tailored {} behaviors to {}, copied news file also".format(
        num, behavior_file))
    return


def expandData():
    """ Beta
    """
    a = pd.read_table(r"D:\Data\MIND\MINDlarge_train\behaviors.tsv",
                      index_col=0, names=["a", "b", "c", "d", "e"], quoting=3)
    b = pd.read_table(r"D:\Data\MIND\MINDlarge_dev\behaviors.tsv",
                      index_col=0, names=["a", "b", "c", "d", "e"], quoting=3)
    c = pd.concat([a, b]).drop_duplicates().reset_index(inplace=True)
    c = c[["b", "c", "d", "e"]]

    c.to_csv(r"D:\Data\MIND\MINDlarge_whole\behaviors.tsv",
             index=True, sep="\t", header=False)


def getId2idx(file):
    """
        get Id2idx dictionary from json file
    """
    g = open(file, "r", encoding="utf-8")
    dic = json.load(g)
    g.close()
    return dic


def getVocab(file):
    """
        get Vocabulary from pkl file
    """
    g = open(file, "rb")
    dic = pickle.load(g)
    g.close()
    return dic


def getLoss(model):
    """
        get loss function for model
    """
    if model.cdd_size > 1:
        if hasattr(model,"contra_num") and model.contra_num:
            loss = myLoss
        else:
            loss = nn.NLLLoss()
    else:
        loss = nn.BCELoss()

    return loss


def getOptim(model, hparams, loader_train):
    """
        get optimizer/scheduler
    """
    if "learning_rate" in hparams:
        learning_rate = hparams["learning_rate"]
    else:
        learning_rate = 1e-3

    if hparams["spadam"]:
        optimizer_param = optim.Adam(
            parameter(model, ["encoder.embedding.weight","encoder.user_embedding.weight"], exclude=True), lr=learning_rate)
        optimizer_embedding = optim.SparseAdam(
            list(parameter(model, ["encoder.embedding.weight","encoder.user_embedding.weight"])), lr=learning_rate)

        optimizers = (optimizer_param, optimizer_embedding)

        if hparams["schedule"] == "linear":
            scheduler_param =get_linear_schedule_with_warmup(optimizer_param, num_warmup_steps=0, num_training_steps=len(loader_train) * hparams["epochs"])
            scheduler_embedding =get_linear_schedule_with_warmup(optimizer_embedding, num_warmup_steps=0, num_training_steps=len(loader_train) * hparams["epochs"])
            schedulers = (scheduler_param, scheduler_embedding)
        else:
            schedulers = []

    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizers = (optimizer,)
        if hparams["schedule"] == "linear":
            scheduler =get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader_train) * hparams["epochs"])
            schedulers = (scheduler,)
        else:
            schedulers = []

    if "checkpoint" in hparams:
        logging.info("loading checkpoint...")
        ck = hparams["checkpoint"].split(",")
        # modify the epoch so the model can be properly saved
        load(model, hparams, ck[0], ck[1], optimizers)

    return optimizers, schedulers


def getLabel(model, x):
    """
        parse labels to label indexes, used in NLLoss
    """
    if model.cdd_size > 1:
        index = torch.arange(0, model.cdd_size, device=model.device).expand(
            model.batch_size, -1)
        label = x["labels"] == 1
        label = index[label]
    else:
        label = x["labels"]

    return label


def save(model, hparams, epoch, step, optimizers=[]):
    """
        shortcut for saving the model and optimizer
    """
    # parse checkpoint
    if "checkpoint" in hparams:
        save_path = "data/model_params/{}/{}_epoch{}_step{}_ck{}_[hs={},topk={},attrs={}].model".format(
            hparams["name"], hparams["scale"], epoch, step, hparams["checkpoint"], hparams["his_size"], hparams["k"], ",".join(hparams["attrs"]))
    else:
        save_path = "data/model_params/{}/{}_epoch{}_step{}_[hs={},topk={},attrs={}].model".format(
            hparams["name"], hparams["scale"], epoch, step, hparams["his_size"], hparams["k"], ",".join(hparams["attrs"]))

    state_dict = model.state_dict()

    if re.search("pipeline", hparams["name"]):
        state_dict = {k: v for k, v in state_dict.items() if k not in [
            "encoder.news_repr.weight", "encoder.news_embedding.weight"]}

    save_dict = {}
    save_dict["model"] = state_dict

    if len(optimizers) > 1:
        save_dict["optimizer"] = optimizers[0].state_dict()
        save_dict["optimizer_embedding"] = optimizers[1].state_dict()
    else:
        save_dict["optimizer"] = optimizers[0].state_dict()

    # if schedulers:
    #     if len(schedulers) > 1:
    #         save_dict["scheduler"] = schedulers[0].state_dict()
    #         save_dict["scheduler_embedding"] = schedulers[1].state_dict()
    #     else:
    #         save_dict["scheduler"] = schedulers[0].state_dict()

    torch.save(save_dict, save_path)
    logging.info("saved model of step {}, epoch {} at {}".format(
        step, epoch, save_path))


def load(model, hparams, epoch, step, optimizers=None):
    """
        shortcut for loading model and optimizer parameters
    """

    if "checkpoint" in hparams:
        save_path = "data/model_params/{}/{}_epoch{}_step{}_ck{}_[hs={},topk={},attrs={}].model".format(
            hparams["name"], hparams["scale"], epoch, step, hparams["checkpoint"], hparams["his_size"], hparams["k"], ",".join(hparams["attrs"]))
    else:
        save_path = "data/model_params/{}/{}_epoch{}_step{}_[hs={},topk={},attrs={}].model".format(
            hparams["name"], hparams["scale"], epoch, step, hparams["his_size"], hparams["k"], ",".join(hparams["attrs"]))

    state_dict = torch.load(save_path, map_location=hparams["device"])
    if re.search("pipeline",model.name):
        logging.info("loading in pipeline")
        model.load_state_dict(state_dict["model"], strict=False)
    else:
        model.load_state_dict(state_dict["model"])

    if optimizers:
        optimizers[0].load_state_dict(state_dict["optimizer"])
        if len(optimizers) > 1:
            optimizers[1].load_state_dict(state_dict["optimizer_embedding"])

    # if schedulers:
    #     schedulers[0].load_state_dict(state_dict["scheduler"])
    #     if len(schedulers) > 1:
    #         schedulers[1].load_state_dict(state_dict["scheduler_embedding"])

    logging.info("Loading model from {}...".format(save_path))


def _log(res, model, hparams):
    """ wrap logging
    """
    logging.info("evaluation results:{}".format(res))
    with open("performance.log", "a+") as f:
        d = {}
        for k, v in hparams.items():
            if k in hparam_list:
                d[k] = v
        if isinstance(model, nn.Module):
            for name, param in model.named_parameters():
                if name in param_list:
                    d[name] = tuple(param.shape)

        f.write(str(d)+"\n")
        f.write(str(res) + "\n")
        f.write("\n")

def my_collate(data):
    excluded = ["impression_index"]
    result = defaultdict(list)
    for d in data:
        for k, v in d.items():
            result[k].append(v)
    for k, v in result.items():
        if k not in excluded:
            result[k] = torch.from_numpy(np.asarray(v))

        else:
            continue
    return dict(result)

# FIXME ugly
def myLoss(pred, label, hidden_dim, margin=1):
    """
        apply contrasive learning for selection-aware model
    """
    log_prob = pred[0]
    cdd_repr = pred[1]
    pos_repr = pred[2]
    neg_repr = pred[3]

    recommend_Loss = nn.NLLLoss()
    select_Loss = TripletMarginLoss(margin=margin)

    reco_loss = recommend_Loss(log_prob, label)
    slct_loss = select_Loss(cdd_repr.unsqueeze(dim=-2).expand(pos_repr.shape).reshape(-1, hidden_dim), pos_repr.view(-1, hidden_dim), neg_repr.view(-1, hidden_dim))

    return reco_loss + slct_loss


def parameter(model, param_list, exclude=False):
    """
        yield model parameters
    """
    # params = []
    if exclude:
        for name, param in model.named_parameters():
            if name not in param_list:
                # params.append(param)
                yield param
    else:
        for name, param in model.named_parameters():
            if name in param_list:
                # params.append(param)
                yield param


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

    for batch_data_input in tqdm(dataloader, smoothing=0.3):
        pred = model(batch_data_input).squeeze(dim=-1).tolist()
        preds.extend(pred)
        label = batch_data_input["labels"].squeeze(dim=-1).tolist()
        labels.extend(label)
        imp_indexes.extend(batch_data_input["impression_index"])

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
def evaluate(model, hparams, dataloader, loading=False, log=True):
    """Evaluate the given file and returns some evaluation metrics.

    Args:
        model(nn.Module)
        hparams(dict)
        dataloader(torch.utils.data.DataLoader): provide data
        load(bool): whether to load model
        interval(int): within each epoch, the interval of steps to display loss

    Returns:
        dict: A dictionary contains evaluation metrics.
    """
    if len(hparams["save_step"]) > 1:
        for step in hparams["save_step"][1:]:
            command = re.sub(",".join([str(i) for i in hparams["save_step"]]), str(
                step), hparams["command"])
            subprocess.Popen(command, shell=True)

    if isinstance(model, nn.Module):
        cdd_size = model.cdd_size
        model.cdd_size = 1
        model.eval()

    if loading:
        load(model, hparams, hparams["epochs"], hparams["save_step"][0])

    logging.info("evaluating...")

    imp_indexes, labels, preds = run_eval(model, dataloader, hparams["interval"])
    res = cal_metric(labels, preds, hparams["metrics"].split(","))

    if log:
        res["epoch"] = hparams["epochs"]
        res["step"] = hparams["save_step"][0]

        _log(res, model, hparams)

    if isinstance(model, nn.Module):
        model.train()
        model.cdd_size = cdd_size

    return res


def run_train(model, dataloader, optimizers, loss_func, hparams, schedulers=None, writer=None, interval=100, save_step=None):
    """ train model and print loss meanwhile
    Args:
        model(torch.nn.Module): the model to be trained
        dataloader(torch.utils.data.DataLoader): provide data
        optimizer(list of torch.nn.optim): optimizer for training
        loss_func(torch.nn.Loss): loss function for training
        hparams(dict): hyper parameters
        writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
        interval(int): within each epoch, the interval of training steps to display loss
        save_epoch(bool): whether to save the model after every epoch
    Returns:
        model: trained model
    """
    total_loss = 0
    total_steps = 0

    for epoch in range(hparams["epochs"]):
        epoch_loss = 0
        tqdm_ = tqdm(dataloader, smoothing=0)
        for step, x in enumerate(tqdm_):

            for optimizer in optimizers:
                optimizer.zero_grad()

            pred = model(x)
            label = getLabel(model, x)

            if hasattr(model, "contra_num") and model.contra_num:
                loss = loss_func(pred, label, model.hidden_dim)
            else:
                loss = loss_func(pred, label)

            epoch_loss += loss
            total_loss += loss

            loss.backward()

            # print(model.selectionProject[0].weight.grad)

            for optimizer in optimizers:
                optimizer.step()

            if schedulers:
                for scheduler in schedulers:
                    scheduler.step()

            if step % interval == 0:

                tqdm_.set_description(
                    "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                if writer:
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param, step)

                    writer.add_scalar("data_loss",
                                      total_loss/total_steps)

            if save_step:
                if step % save_step == 0 and step > 0:
                    save(model, hparams, epoch+1, step, optimizers)

            total_steps += 1

        if writer:
            writer.add_scalar("epoch_loss", epoch_loss, epoch)

        save(model, hparams, epoch+1, 0, optimizers)

    return model


def train(model, hparams, loaders, tb=False):
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
        writer = SummaryWriter("data/tb/{}/{}/{}/".format(
            hparams["name"], hparams["scale"], datetime.now().strftime("%Y%m%d-%H")))

    # in case the folder does not exists, create one
    save_derectory = "data/model_params/{}".format(hparams["name"])
    if not os.path.exists(save_derectory):
        os.mkdir(save_derectory)

    logging.info("training...")
    loss_func = getLoss(model)
    optimizers, schedulers = getOptim(model, hparams, loaders[0])

    model = run_train(model, loaders[0], optimizers, loss_func, hparams, schedulers=schedulers,
                      writer=writer, interval=hparams["interval"], save_step=hparams["save_step"][0])

    # loader_train, loader_dev, loader_validate
    # if len(loaders) > 1:
    #     for loader in loaders[1:]:
    #         evaluate(model, hparams, loader)

    return model


def run_tune(model, loaders, optimizers, loss_func, hparams, schedulers=[], writer=None, interval=100, save_step=None):
    """ train model and print loss meanwhile
    Args:
        model(torch.nn.Module): the model to be trained
        dataloader(torch.utils.data.DataLoader): provide data
        optimizer(list of torch.nn.optim): optimizer for training
        loss_func(torch.nn.Loss): loss function for training
        hparams(dict): hyper parameters
        writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
        interval(int): within each epoch, the interval of training steps to display loss
        save_epoch(bool): whether to save the model after every epoch
    Returns:
        model: trained model
    """
    total_loss = 0
    total_steps = 0

    best_res = {"auc":0}

    for epoch in range(hparams["epochs"]):
        epoch_loss = 0
        tqdm_ = tqdm(loaders[0], smoothing=0)
        for step, x in enumerate(tqdm_):

            for optimizer in optimizers:
                optimizer.zero_grad()

            pred = model(x)
            label = getLabel(model, x)

            if hasattr(model, "contra_num") and model.contra_num:
                loss = loss_func(pred, label, model.hidden_dim)
            else:
                loss = loss_func(pred, label)

            epoch_loss += loss
            total_loss += loss

            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            if schedulers:
                for scheduler in schedulers:
                    scheduler.step()

            if step % interval == 0:

                tqdm_.set_description(
                    "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                if writer:
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param, step)

                    writer.add_scalar("data_loss",
                                      total_loss/total_steps)

            if step % save_step == 0 and step > 0:
                print("\n")
                result = evaluate(model, hparams, loaders[1], log=False)
                result["epoch"] = epoch+1
                result["step"] = step

                logging.info("current result of {} is {}".format(hparams["name"], result))
                if result["auc"] > best_res["auc"]:
                    best_res = result
                    logging.info("best result till now is {}".format(best_res))
                    save(model, hparams, epoch+1, step, optimizers)
                    _log(result, model, hparams)

                elif result["auc"] - best_res["auc"] < -0.05:
                    logging.info("model is overfitting, the result is {}, force shutdown".format(result))
                    return model, best_res

            total_steps += 1

        if writer:
            writer.add_scalar("epoch_loss", epoch_loss, epoch)

    return model, best_res


def tune(model, hparams, loaders, tb=False):
    """ train and evaluate sequentially

    Args:
        model(torch.nn.Module): the model to be trained
        loaders(list): list of torch.utils.data.DataLoader
        hparams(dict): hyper paramaters
        en: shell parameter
    """

    model.train()
    writer = None

    if tb:
        writer = SummaryWriter("data/tb/{}/{}/{}/".format(
            hparams["name"], hparams["scale"], datetime.now().strftime("%Y%m%d-%H")))

    # in case the folder does not exists, create one
    save_derectory = "data/model_params/{}".format(hparams["name"])
    if not os.path.exists(save_derectory):
        os.mkdir(save_derectory)

    logging.info("training...")
    loss_func = getLoss(model)
    optimizers, schedulers = getOptim(model, hparams, loaders[0])

    model, res = run_tune(model, loaders, optimizers, loss_func, hparams, schedulers=schedulers,
                      writer=writer, interval=hparams["interval"], save_step=int(len(loaders[0])/hparams["val_freq"])-1)

    _log(res, model, hparams)
    return model


@torch.no_grad()
def test(model, hparams, loader_test):
    """ test the model on test dataset of MINDlarge

    Args:
        model
        hparams
        loader_test: DataLoader of MINDlarge_test
    """
    load(model, hparams, hparams["epochs"], hparams["save_step"][0])

    logging.info("testing...")
    try:
        model.cdd_size = 1
        model.eval()
    except:
        logging.info("this model is not inherited from nn.Module")

    with open("performance.log", "a+") as f:
        d = {}
        for k, v in hparams.items():
            if k in hparam_list:
                d[k] = v
        for name, param in model.named_parameters():
            if name in param_list:
                d[name] = tuple(param.shape)

        f.write(str(d)+"\n")
        f.write("\n")
        f.write("\n")

    save_path = "data/results/prediction={}_{}_epoch{}_step{}_[hs={},topk={}].txt".format(
        hparams["name"], hparams["scale"], hparams["epochs"], hparams["save_step"][0], hparams["his_size"], hparams["k"])
    with open(save_path, "w") as f:
        preds = []
        imp_indexes = []
        for x in tqdm(loader_test, smoothing=0):
            preds.extend(model.forward(x).tolist())
            imp_indexes.extend(x["impression_index"])

        all_keys = list(set(imp_indexes))
        all_keys.sort()
        group_preds = {k: [] for k in all_keys}

        for i, p in zip(imp_indexes, preds):
            group_preds[i].append(p)

        for k, v in group_preds.items():
            array = np.asarray(v)
            rank_list = ss.rankdata(1 - array, method="ordinal")
            line = str(k) + " [" + ",".join([str(i)
                                             for i in rank_list]) + "]" + "\n"
            f.write(line)

    logging.info("written to prediction at {}!".format(save_path))


def load_hparams(hparams):
    """
        customize hyper parameters in command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scale", dest="scale", help="data scale",
                        choices=["demo", "small", "large", "whole"], required=True)
    parser.add_argument("-m", "--mode", dest="mode", help="train or test",
                        choices=["train", "dev", "test", "tune", "encode"], default="train")
    parser.add_argument("-e", "--epochs", dest="epochs",
                        help="epochs to train the model", type=int, default=10)

    parser.add_argument("-bs", "--batch_size", dest="batch_size",
                        help="batch size", type=int, default=100)
    parser.add_argument("-ts", "--title_size", dest="title_size",
                        help="news title size", type=int, default=20)
    parser.add_argument("--abs_size", dest="abs_size",
                        help="news abstract length", type=int, default=40)
    parser.add_argument("-hs", "--his_size", dest="his_size",
                        help="history size", type=int, default=50)

    parser.add_argument("--device", dest="device",
                        help="device to run on", choices=["0", "1", "cpu"], default="0")
    parser.add_argument("--interval", dest="interval", help="the step interval to update processing bar", default=0, type=int)
    parser.add_argument("--save_step", dest="save_step",
                        help="if clarified, save model at the interval of given steps", type=str, default="0")
    parser.add_argument("--val_freq", dest="val_freq", help="the frequency to validate during training in one epoch", type=int, default=0)

    parser.add_argument("-ck", "--checkpoint", dest="checkpoint",
                        help="the checkpoint model to load", type=str)
    parser.add_argument("--learning_rate", dest="learning_rate",
                        help="learning rate when training", type=float, default=0)
    parser.add_argument("--schedule", dest="schedule", help="choose schedule scheme for optimizer", default="linear")

    parser.add_argument("--npratio", dest="npratio",
                        help="the number of unclicked news to sample when training", type=int, default=4)
    parser.add_argument("-mc", "--metrics", dest="metrics",
                        help="metrics for evaluating the model, if multiple metrics are needed, seperate with ','", type=str, default="auc,mean_mrr,ndcg@5,ndcg@10")

    parser.add_argument(
        "--topk", dest="k", help="intend for sfi model, if clarified, top k history are involved in interaction calculation", type=int, default=0)
    parser.add_argument(
        "--contra_num", dest="contra_num", help="sample number for contrasive selection aware network", type=int, default=0)
    parser.add_argument("--coarse", dest="coarse", help="if clarified, coarse-level matching signals will be taken into consideration",action='store_true')
    parser.add_argument("--integrate", dest="integration",
                        help="the way history filter is combined", choices=["gate", "harmony"], default="gate")
    parser.add_argument("--encoder", dest="encoder", help="choose encoder", default="fim")
    parser.add_argument("--interactor", dest="interactor", help="choose interactor", default="fim")
    parser.add_argument("--threshold", dest="threshold", help="if clarified, SFI will dynamically mask attention weights smaller than threshold with 0", default=-float("inf"), type=float)
    parser.add_argument("--multiview", dest="multiview", help="if clarified, SFI-MultiView will be called", action="store_true")
    parser.add_argument("--ensemble", dest="ensemble", help="choose ensemble strategy for SFI-ensemble", type=str, default=None)


    parser.add_argument("--bert", dest="bert", help="choose bert model(encoder)",
                        choices=["bert-base-uncased", "albert-base-v2"], default=None)
    parser.add_argument("--level", dest="level",
                        help="intend for bert encoder, if clarified, level representations will be kept for a token", type=int, default=1)

    # FIXME, clarify all choices
    parser.add_argument("--pipeline", dest="pipeline", help="choose pipeline-encoder", default=None)

    parser.add_argument("-hn", "--head_num", dest="head_num",
                        help="number of multi-heads", type=int, default=16)
    parser.add_argument("-vd", "--value_dim", dest="value_dim",
                        help="dimension of projected value", type=int, default=16)
    parser.add_argument("-qd", "--query_dim", dest="query_dim",
                        help="dimension of projected query", type=int, default=200)

    parser.add_argument("--attrs", dest="attrs",
                        help="clarified attributes of news will be yielded by dataloader, seperate with comma", type=str, default="title")
    parser.add_argument("--validate", dest="validate",
                        help="if clarified, evaluate the model on training set", action="store_true")
    parser.add_argument("--onehot", dest="onehot", help="if clarified, one hot encode of category/subcategory will be returned by dataloader", action="store_true")

    args = parser.parse_args()

    hparams["scale"] = args.scale
    hparams["mode"] = args.mode
    if hparams["mode"] == "train":
        # 2000 by default
        hparams["save_step"] = 2000
    if len(args.device) > 1:
        hparams["device"] = args.device
    else:
        hparams["device"] = "cuda:" + args.device
    hparams["epochs"] = args.epochs
    hparams["batch_size"] = args.batch_size
    hparams["interval"] = args.interval
    hparams["title_size"] = args.title_size
    hparams["abs_size"] = args.abs_size
    hparams["npratio"] = args.npratio
    hparams["metrics"] = args.metrics
    hparams["val_freq"] = args.val_freq
    hparams["schedule"] = args.schedule
    hparams["spadam"] = True
    hparams["contra_num"] = args.contra_num
    hparams["head_num"] = args.head_num
    hparams["value_dim"] = args.value_dim
    hparams["query_dim"] = args.query_dim
    hparams["interactor"] = args.interactor

    hparams["his_size"] = args.his_size
    hparams["k"] = args.k

    hparams["threshold"] = args.threshold

    hparams["attrs"] = args.attrs.split(",")
    hparams["save_step"] = [int(i) for i in args.save_step.split(",")]

    if not args.learning_rate:
        if hparams["scale"] == "demo":
            hparams["learning_rate"] = 1e-3
        else:
            hparams["learning_rate"] = 1e-4
    else:
        hparams["learning_rate"] = args.learning_rate
    if not args.interval:
        if hparams["scale"] == "demo":
            hparams["interval"] = 10
        else:
            hparams["interval"] = 100
    else:
        hparams["interval"] = args.interval
    if not args.val_freq:
        if hparams["scale"] == "demo":
            hparams["val_freq"] = 1
        else:
            hparams["val_freq"] = 4
    else:
        hparams["val_freq"] = args.val_freq
    if args.validate:
        hparams["validate"] = args.validate
    if args.onehot:
        hparams["onehot"] = args.onehot
        hparams["vert_num"] = 18
        hparams["subvert_num"] = 293
    else:
        hparams["onehot"] = False
    if args.checkpoint:
        hparams["checkpoint"] = args.checkpoint
    if args.encoder:
        hparams["encoder"] = args.encoder
    if args.multiview:
        hparams["multiview"] = args.multiview
        hparams["attrs"] = "title,vert,subvert,abs".split(",")
        logging.info("automatically set True for onehot encoding of (sub)categories")
        hparams["onehot"] = True
        hparams["vert_num"] = 18
        hparams["subvert_num"] = 293
    else:
        hparams["multiview"] = False
    if args.coarse:
        hparams['coarse'] = 'coarse'
    else:
        hparams['coarse'] = None
    if args.ensemble:
        hparams["ensemble"] = args.ensemble
    if args.coarse:
        hparams["integration"] = args.integration
        hparams["coarse"] = "coarse"
    if args.pipeline:
        hparams["pipeline"] = args.pipeline
        hparams["encoder"] = "pipeline"
        hparams["name"] = args.pipeline
        hparams["spadam"] = False
    if args.bert:
        hparams["encoder"] = "bert"
        hparams["bert"] = args.bert
        hparams["level"] = args.level

    if args.k < 5 and args.k > 0:
        logging.warning("k should always be larger than 4")
        hparams['k'] = 5

    if len(hparams["save_step"]) > 1:
        hparams["command"] = "python " + " ".join(sys.argv)

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


def prepare(hparams, path="/home/peitian_zhang/Data/MIND", shuffle=True, news=False, pin_memory=True, num_workers=8):
    from .MIND import MIND,MIND_news,MIND_all
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
        path = "/home/peitian_zhang/Data/MIND"
        news_file_train = path + \
            "/MIND{}_train/news.tsv".format(hparams["scale"])
        news_file_dev = path+"/MIND{}_dev/news.tsv".format(hparams["scale"])

        dataset_train = MIND_news(hparams, news_file_train)
        loader_news_train = DataLoader(
            dataset_train, batch_size=hparams["batch_size"], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, collate_fn=my_collate)

        dataset_dev = MIND_news(hparams, news_file_dev)
        loader_news_dev = DataLoader(
            dataset_dev, batch_size=hparams["batch_size"], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, collate_fn=my_collate)

        vocab = dataset_train.vocab
        embedding = GloVe(dim=300, cache=".vector_cache")
        vocab.load_vectors(embedding)

        if hparams["scale"] == "large":
            news_file_test = path + \
                "/MIND{}_test/news.tsv".format(hparams["scale"])
            dataset_test = MIND_news(hparams, news_file_test)
            loader_news_test = DataLoader(
                dataset_test, batch_size=hparams["batch_size"], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, collate_fn=my_collate)

            return vocab, [loader_news_train, loader_news_dev, loader_news_test]

        return vocab, [loader_news_train, loader_news_dev]

    elif hparams["mode"] in ["train", "tune"]:
        news_file_train = path+"/MIND"+hparams["scale"]+"_train/news.tsv"
        behavior_file_train = path+"/MIND" + \
            hparams["scale"]+"_train/behaviors.tsv"
        news_file_dev = path+"/MIND"+hparams["scale"]+"_dev/news.tsv"
        behavior_file_dev = path+"/MIND"+hparams["scale"]+"_dev/behaviors.tsv"

        if hparams["multiview"]:
            dataset_train = MIND_all(hparams=hparams, news_file=news_file_train,
                            behaviors_file=behavior_file_train)
            dataset_dev = MIND_all(hparams=hparams, news_file=news_file_dev,
                            behaviors_file=behavior_file_dev)
        else:
            dataset_train = MIND(hparams=hparams, news_file=news_file_train,
                                behaviors_file=behavior_file_train)
            dataset_dev = MIND(hparams=hparams, news_file=news_file_dev,
                            behaviors_file=behavior_file_dev)

        vocab = dataset_train.vocab
        if "bert" not in hparams:
            embedding = GloVe(dim=300, cache=".vector_cache")
            vocab.load_vectors(embedding)
        loader_train = DataLoader(dataset_train, batch_size=hparams["batch_size"], pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle, collate_fn=my_collate)
        loader_dev = DataLoader(dataset_dev, batch_size=hparams["batch_size"], pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, collate_fn=my_collate)

        if "validate" in hparams and hparams["validate"]:
            if hparams["multiview"]:
                dataset_validate = MIND_all(
                    hparams=hparams, news_file=news_file_train, behaviors_file=behavior_file_train, validate=True)
            else:
                dataset_validate = MIND(
                    hparams=hparams, news_file=news_file_train, behaviors_file=behavior_file_train, validate=True)
            loader_validate = DataLoader(dataset_validate, batch_size=hparams["batch_size"], pin_memory=pin_memory,
                                        num_workers=num_workers, drop_last=False, collate_fn=my_collate)
            return vocab, [loader_train, loader_dev, loader_validate]
        else:
            return vocab, [loader_train, loader_dev]

    elif hparams["mode"] == "dev":
        news_file_dev = path+"/MIND"+hparams["scale"]+"_dev/news.tsv"
        behavior_file_dev = path+"/MIND"+hparams["scale"]+"_dev/behaviors.tsv"

        if hparams["multiview"]:
            dataset_dev = MIND_all(hparams=hparams, news_file=news_file_dev,
                            behaviors_file=behavior_file_dev)
        else:
            dataset_dev = MIND(hparams=hparams, news_file=news_file_dev,
                            behaviors_file=behavior_file_dev)
        loader_dev = DataLoader(dataset_dev, batch_size=hparams["batch_size"], pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, collate_fn=my_collate)
        vocab = dataset_dev.vocab
        if "bert" not in hparams:
            embedding = GloVe(dim=300, cache=".vector_cache")
            vocab.load_vectors(embedding)

        return vocab, [loader_dev]

    elif hparams["mode"] == "test":
        if hparams["multiview"]:
            dataset_test = MIND_all(hparams, "/home/peitian_zhang/Data/MIND/MINDlarge_test/news.tsv",
                                    "/home/peitian_zhang/Data/MIND/MINDlarge_test/behaviors.tsv")
        else:
            dataset_test = MIND(hparams, "/home/peitian_zhang/Data/MIND/MINDlarge_test/news.tsv",
                                    "/home/peitian_zhang/Data/MIND/MINDlarge_test/behaviors.tsv")
        loader_test = DataLoader(dataset_test, batch_size=hparams["batch_size"], pin_memory=pin_memory,
                                 num_workers=num_workers, drop_last=False, collate_fn=my_collate)
        vocab = dataset_test.vocab
        if "bert" not in hparams:
            embedding = GloVe(dim=300, cache=".vector_cache")
            vocab.load_vectors(embedding)

        return vocab, [loader_test]


def pipeline_encode(model, hparams, loaders):
    """
        Encode news of hparams["scale"] in each mode
    """
    news_num_dict = {
        "demo": {
            "train": 51282,
            "dev": 42416
        },
        "small": {
            "train": 51282,
            "dev": 42416
        },
        "large": {
            "train": 101527,
            "dev": 72023,
            "test": 120961
        }
    }
    news_num = news_num_dict[hparams["scale"]]["train"]

    news_reprs = torch.zeros((news_num + 1, model.hidden_dim))
    news_embeddings = torch.zeros(
        (news_num + 1, model.signal_length, model.level, model.hidden_dim))

    for x in tqdm(loaders[0]):
        embedding, repr = model(x)
        for i in range(embedding.shape[0]):
            news_reprs[x["news_id"][i]] = repr[i]
            news_embeddings[x["news_id"][i]] = embedding[i]

    torch.save(news_reprs, "data/tensors/news_repr_{}_train-[{}].tensor".format(
        hparams["scale"], hparams["name"]))
    torch.save(news_embeddings, "data/tensors/news_embedding_{}_train-[{}].tensor".format(
        hparams["scale"], hparams["name"]))
    del news_reprs
    del news_embeddings

    news_num_dev = news_num_dict[hparams["scale"]]["dev"]

    news_reprs = torch.zeros((news_num_dev + 1, model.hidden_dim))
    news_embeddings = torch.zeros(
        (news_num_dev + 1, model.signal_length, model.level, model.hidden_dim))

    for x in tqdm(loaders[1]):
        embedding, repr = model(x)
        for i in range(embedding.shape[0]):
            news_reprs[x["news_id"][i]] = repr[i]
            news_embeddings[x["news_id"][i]] = embedding[i]

    torch.save(news_reprs, "data/tensors/news_repr_{}_dev-[{}].tensor".format(
        hparams["scale"], hparams["name"]))
    torch.save(news_embeddings, "data/tensors/news_embedding_{}_dev-[{}].tensor".format(
        hparams["scale"], hparams["name"]))
    del news_reprs
    del news_embeddings


    if hparams["scale"] == "large":
        news_num_test = news_num_dict[hparams["scale"]]["test"]

        news_reprs = torch.zeros((news_num_test + 1, model.hidden_dim))
        news_embeddings = torch.zeros(
            (news_num_test + 1, model.signal_length, model.level, model.hidden_dim))

        for x in tqdm(loaders[2]):
            embedding, repr = model(x)
            for i in range(embedding.shape[0]):
                news_reprs[x["news_id"][i]] = repr[i]
                news_embeddings[x["news_id"][i]] = embedding[i]

        torch.save(news_reprs, "data/tensors/news_repr_{}_test-[{}].tensor".format(
            hparams["scale"], hparams["name"]))
        torch.save(news_embeddings, "data/tensors/news_embedding_{}_test-[{}].tensor".format(
            hparams["scale"], hparams["name"]))
        del news_reprs
        del news_embeddings

    logging.info("successfully encoded news!")

@torch.no_grad()
def encode(model, hparams, loader=None):
    """
        Encode news of hparams["scale"] in each mode, currently force to encode dev dataset
    """

    # very important
    model.eval()

    if not loader:
        from .MIND import MIND_news
        path = "/home/peitian_zhang/Data/MIND"
        news_file = path + "/MIND{}_{}/news.tsv".format(hparams["scale"],hparams["mode"])

        dataset = MIND_news(hparams, news_file)
        loader_news = DataLoader(
            dataset, batch_size=hparams["batch_size"], pin_memory=False, num_workers=8, drop_last=False, collate_fn=my_collate)

    news_num_dict = {
        "demo": {
            "train": 51282,
            "dev": 42416
        },
        "small": {
            "train": 51282,
            "dev": 42416
        },
        "large": {
            "train": 101527,
            "dev": 72023,
            "test": 120961
        }
    }
    news_num = news_num_dict[hparams["scale"]]["dev"]

    news_reprs = torch.zeros((news_num + 1, model.hidden_dim))
    news_embeddings = torch.zeros(
        (news_num + 1, model.signal_length, model.level, model.hidden_dim))

    for x in tqdm(loader):
        embedding, repr = model(x)
        for i in range(embedding.shape[0]):
            news_reprs[x["cdd_id"][i]] = repr[i]
            news_embeddings[x["cdd_id"][i]] = embedding[i]

    torch.save(news_reprs, "data/tensors/news_repr_{}_{}-[{}].tensor".format(
        hparams["scale"], "dev", hparams["name"]))
    torch.save(news_embeddings, "data/tensors/news_embedding_{}_{}-[{}].tensor".format(
        hparams["scale"], "dev", hparams["name"]))
    del news_reprs
    del news_embeddings
    logging.info("successfully encoded news of {}-{}, saved in data/tensors/news_**_{}_{}-[{}].tensor".format(hparams["scale"], "dev", hparams["scale"], "dev", hparams["name"]))

def analyse(hparams, path="/home/peitian_zhang/Data/MIND"):
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
        "/MIND{}_{}/news.tsv".format(hparams["scale"], hparams["mode"])

    behavior_file = path + \
        "/MIND{}_{}/behaviors.tsv".format(hparams["scale"], hparams["mode"])

    with open(news_file, "r", encoding="utf-8") as rd:
        count = 0
        for idx in rd:
            nid, vert, subvert, title, ab, url, _, _ = idx.strip(
                "\n").split("\t")
            avg_title_length += len(title.split(" "))
            avg_abstract_length += len(ab.split(" "))
            count += 1
    avg_title_length = avg_title_length/count
    avg_abstract_length = avg_abstract_length/count

    with open(behavior_file, "r", encoding="utf-8") as rd:
        count = 0
        for idx in rd:
            uid, time, history, impr = idx.strip("\n").split("\t")[-4:]
            his = history.split(" ")
            imp = impr.split(" ")
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
