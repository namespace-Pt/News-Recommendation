import re
import os
import math
import pickle
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import sample
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score



def load_pickle(path):
    """ load pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def download_plm(bert, dir):
    # initialize bert related parameters
    bert_loading_map = {
        "bert": "bert-base-uncased",
        "deberta": "microsoft/deberta-base",
    }
    os.makedirs(dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bert_loading_map[bert])
    model = AutoModel.from_pretrained(bert_loading_map[bert])
    tokenizer.save_pretrained(dir)
    model.save_pretrained(dir)


def pack_results(impr_indices, masks, *associated_lists):
        """
            group lists by impr_index
        Args:
            associated_lists: list of lists, where list[i] is associated with the impr_indices[i]

        Returns:
            Iterable: grouped labels (if inputted) and preds
        """
        list_num = len(associated_lists)
        dicts = [defaultdict(list) for i in range(list_num)]

        for x in tqdm(zip(impr_indices, masks, *associated_lists), total=len(impr_indices), desc="Packing Results", ncols=80):
            key = x[0]
            mask = x[1]
            values = x[2:]
            for i in range(list_num):
                dicts[i][key].extend(values[i][mask].tolist())

        grouped_lists = [list(d.values()) for d in dicts]
        return grouped_lists


def sample_news(news, k):
    """ Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
        int: count of valid news
    """
    num = len(news)
    if k > num:
        return news + [0] * (k - num), num
    else:
        return sample(news, k), k


def tokenize(sent):
    """ Split sentence into words
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[-\w_]+|[.,!?;|]")

    return [x for x in pat.findall(sent.lower())]


def construct_nid2index(news_path, cache_dir):
    """
        Construct news ID to news INDEX dictionary, index starting from 1
    """
    news_df = pd.read_table(news_path, index_col=None, names=[
                            "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3)

    nid2index = {}
    for v in news_df["newsID"]:
        if v in nid2index:
            continue
        # plus one because all news offsets from 1
        nid2index[v] = len(nid2index) + 1
    save_pickle(nid2index, os.path.join(cache_dir, "nid2index.pkl"))


def construct_uid2index(data_root, cache_root):
    """
        Construct user ID to user IDX dictionary, index starting from 0
    """
    uid2index = {}
    user_df_list = []
    behaviors_file_list = [os.path.join(data_root, "MIND", directory, "behaviors.tsv") for directory in ["MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"]]

    for f in behaviors_file_list:
        user_df_list.append(pd.read_table(f, index_col=None, names=[
                            "imprID", "uid", "time", "hisstory", "abstract", "impression"], quoting=3)["uid"])
    user_df = pd.concat(user_df_list).drop_duplicates()
    for v in user_df:
        uid2index[v] = len(uid2index)
    save_pickle(uid2index, os.path.join(cache_root, "MIND", "uid2index.pkl"))
    return uid2index


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


def compute_metrics(labels, preds, metrics):
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


class Sequential_Sampler:
    def __init__(self, dataset_length, num_replicas, rank) -> None:
        super().__init__()
        len_per_worker = dataset_length / num_replicas
        self.start = round(len_per_worker * rank)
        self.end = round(len_per_worker * (rank + 1))

    def __iter__(self):
        start = self.start
        end = self.end
        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start



class BM25(object):
    """
    compute bm25 score on the entire corpus, instead of the one limited by signal_length
    """
    def __init__(self, k=0.9, b=0.4):
        self.k = k
        self.b = b
        self.logger = logging.getLogger("BM25")


    def fit(self, documents):
        """
        build term frequencies (how many times a term occurs in one news) and document frequencies (how many documents contains a term)
        """
        doc_length = 0
        doc_count = len(documents)

        tfs = []
        df = defaultdict(int)
        for document in documents:
            tf = defaultdict(int)
            words = tokenize(document)
            for word in words:
                tf[word] += 1
                df[word] += 1
            tfs.append(tf)
            doc_length += len(words)

        self.tfs = tfs

        idf = defaultdict(float)
        for word, freq in df.items():
            idf[word] = math.log((doc_count - freq + 0.5 ) / (freq + 0.5) + 1)

        self.idf = idf
        self.doc_avg_length = doc_length / doc_count


    def __call__(self, documents):
        self.logger.info("computing BM25 scores...")
        if not hasattr(self, "idf"):
            self.fit(documents)
        sorted_documents = []
        for tf, document in zip(self.tfs, documents):
            score_pairs = []
            for word, freq in tf.items():
                # skip word such as punctuations
                if len(word) == 1:
                    continue
                score = (self.idf[word] * freq * (self.k + 1)) / (freq + self.k * (1 - self.b + self.b * len(document) / self.doc_avg_length))
                score_pairs.append((word, score))
            score_pairs = sorted(score_pairs, key=lambda x: x[1], reverse=True)
            sorted_document = " ".join([x[0] for x in score_pairs])
            sorted_documents.append(sorted_document)
        return sorted_documents
