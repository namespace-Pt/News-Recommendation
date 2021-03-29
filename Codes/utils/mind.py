import numpy as np
import re
from transformers import AutoTokenizer
from torch.utils.data import Dataset, IterableDataset
from utils.utils import newsample, getId2idx, word_tokenize_vocab, getVocab

class MIND(Dataset):
    """ Map Style Dataset for MIND, return positive samples with negative sampling when training, or return each sample when developing.

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, hparams, news_file, behaviors_file, shuffle_pos=False):
        # initiate the whole iterator
        self.npratio = hparams['npratio']
        self.shuffle_pos = shuffle_pos

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.attrs = hparams['attrs']
        self.k = hparams['k']

        self.mode = re.search(
            '{}_(.*)/'.format(hparams['scale']), news_file).group(1)

        self.vocab = getVocab(
            'data/dictionaries/vocab_{}.pkl'.format('_'.join(hparams['attrs'])))
        self.nid2index = getId2idx(
            'data/dictionaries/nid2idx_{}_{}.json'.format(hparams['scale'], self.mode))
        self.uid2index = getId2idx(
            'data/dictionaries/uid2idx_{}.json'.format(hparams['scale']))

        self.bert = False
        if 'bert' in hparams:
            self.bert = True
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['bert'])

        self.init_news()
        self.init_behaviors()

    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME 
        # The nid2idx dictionary must follow the original order of news in news.tsv

        title_token = [[0]*self.title_size]
        title_pad = [[self.title_size]]
        # pure text of the title
        titles = [['hello MIND']]

        with open(self.news_file, "r", encoding='utf-8') as rd:

            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(
                    self.col_spliter
                )

                titles.append([title])
                title = word_tokenize_vocab(title, self.vocab)

                title_token.append(
                    title[:self.title_size] + [1] * (self.title_size - len(title)))
                title_pad.append([max(self.title_size - len(title), 0)])

        self.titles = titles
        self.news_title_array = np.asarray(title_token)
        self.title_pad = np.asarray(title_pad)

    def init_behaviors(self):
        """
            init behavior logs given behaviors file.
        """
        # list of list of history news index
        self.histories = []
        # list of user index
        self.uindexes = []
        # list of list of history padding length
        self.his_pad = []
        # list of impression indexes
        # self.impr_indexes = []

        # only store positive behavior
        if self.mode == 'train':
            # list of list of clicked candidate news index along with its impression index
            self.imprs = []
            # dictionary of list of unclicked candidate news index
            self.negtives = {}

            with open(self.behaviors_file, "r", encoding='utf-8') as rd:
                for idx in rd:
                    impr_index, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)
                    # important to subtract 1 because all list related to behaviors start from 0
                    impr_index = int(impr_index) - 1

                    history = [self.nid2index[i] for i in history.split()]
                    if self.k:
                        # guarantee there are at least k history not masked
                        self.his_pad.append(
                            min(max(self.his_size - len(history), 0), self.his_size - self.k))
                    else:
                        self.his_pad.append(max(self.his_size - len(history), 0))

                    # tailor user's history or pad 0
                    history = history[:self.his_size] + [0] * (self.his_size - len(history))
                    impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                    labels = [int(i.split("-")[1]) for i in impr.split()]
                    # user will always in uid2index
                    uindex = self.uid2index[uid]

                    # store negative samples of each impression
                    negatives = []

                    for news, label in zip(impr_news, labels):
                        if label == 1:
                            self.imprs.append((impr_index, news))
                        else:
                            negatives.append(news)

                    # 1 impression correspond to 1 of each of the following properties
                    self.histories.append(history)
                    self.negtives[impr_index] = negatives
                    self.uindexes.append(uindex)

        # store every behaviors
        elif self.mode == 'dev':
            # list of every candidate news index along with its impression index and label
            self.imprs = []

            with open(self.behaviors_file, "r", encoding='utf-8') as rd:
                for idx in rd:
                    impr_index, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)
                    impr_index = int(impr_index) - 1

                    history = [self.nid2index[i] for i in history.split()]
                    if self.k:
                        # guarantee there are at least k history not masked
                        self.his_pad.append(
                            min(max(self.his_size - len(history), 0), self.his_size - self.k))
                    else:
                        self.his_pad.append(max(self.his_size - len(history), 0))

                    # tailor user's history or pad 0
                    history = history[:self.his_size] + [0] * (self.his_size - len(history))
                    impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                    labels = [int(i.split("-")[1]) for i in impr.split()]
                    # user will always in uid2index
                    uindex = self.uid2index[uid]

                    # store every impression
                    for news, label in zip(impr_news, labels):
                        self.imprs.append((impr_index, news, label))

                    # 1 impression correspond to 1 of each of the following properties
                    self.histories.append(history)
                    self.uindexes.append(uindex)

    def __len__(self):
        """
            return length of the whole dataset
        """
        return len(self.imprs)

    def __getitem__(self,index):
        """ return data
        Args:
            index: the index for stored impression

        Returns:
            back_dic: dictionary of data slice
        """

        impr = self.imprs[index] # (impression_index, news_index)
        impr_index = impr[0]
        impr_news = impr[1]

        user_index = [self.uindexes[impr_index]]

        # each time called to return positive one sample and its negative samples
        if self.mode == 'train':
            # user's unclicked news in the same impression
            negs = self.negtives[impr_index]
            neg_list, neg_pad = newsample(negs, self.npratio)

            cdd_ids = np.asarray([impr_news] + neg_list)
            label = np.asarray([1] + [0] * self.npratio)

            if self.shuffle_pos:
                s = np.arange(0, len(label), 1)
                np.random.shuffle(s)
                cdd_ids = np.asarray(cdd_ids)[s]
                label = np.asarray(label)[s]

            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1), dtype=bool)
            his_ids = self.histories[impr_index]

            # in case the user has no history records, do not mask
            if self.his_pad[impr_index] == self.his_size or self.his_pad[impr_index] == 0:
                his_mask = his_mask
            else:
                his_mask[-self.his_pad[impr_index]:] = [True]

            # pad in candidate
            # candidate_mask = [1] * neg_pad + [0] * (self.npratio + 1 - neg_pad)

            if self.bert:
                encoded_title = self.tokenizer([self.titles[i] for i in cdd_ids],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                candidate_title_index = encoded_title['input_ids']
                candidate_attn_mask = encoded_title['attention_mask']

                encoded_title = self.tokenizer([self.titles[i] for i in his_ids],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                clicked_title_index = encoded_title['input_ids']
                clicked_attn_mask = encoded_title['attention_mask']

                back_dic = {
                    "user_index": np.asarray(user_index),
                    # "cdd_mask": np.asarray(neg_pad),
                    'cdd_id': cdd_ids,
                    "candidate_title": candidate_title_index,
                    "candidate_title_pad": candidate_attn_mask,
                    'his_id': np.asarray(his_ids),
                    "clicked_title": clicked_title_index,
                    "clicked_title_pad": clicked_attn_mask,
                    "his_mask": his_mask,
                    "labels": label
                }

            else:
                # pad in title
                candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
                click_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]

                candidate_title_index = self.news_title_array[cdd_ids]
                clicked_title_index = self.news_title_array[his_ids]

                back_dic = {
                    "user_index": np.asarray(user_index),
                    # "cdd_mask": np.asarray(neg_pad),
                    'cdd_id': cdd_ids,
                    "candidate_title": candidate_title_index,
                    "candidate_title_pad": np.asarray(candidate_title_pad),
                    'his_id': np.asarray(his_ids),
                    "clicked_title": clicked_title_index,
                    "clicked_title_pad": np.asarray(click_title_pad),
                    "his_mask": his_mask,
                    "labels": label
                }

            return back_dic
        
        # each time called return one sample
        elif self.mode == 'dev':
            cdd_ids = [impr_news]

            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1), dtype=bool)
            his_ids = self.histories[impr_index]

            user_index = [self.uindexes[impr_index]]
            label = impr[2]

            # in case the user has no history records, do not mask
            if self.his_pad[impr_index] == self.his_size or self.his_pad[impr_index] == 0:
                his_mask = his_mask
            else:
                his_mask[-self.his_pad[impr_index]:] = [True]

            if self.bert:
                encoded_title = self.tokenizer(self.titles[impr_news],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                candidate_title_index = encoded_title['input_ids']
                candidate_attn_mask = encoded_title['attention_mask']

                encoded_title = self.tokenizer([self.titles[i] for i in his_ids],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                clicked_title_index = encoded_title['input_ids']
                clicked_attn_mask = encoded_title['attention_mask']

                back_dic = {
                    "impression_index":impr_index,
                    "user_index": np.asarray(user_index),
                    'cdd_id': np.asarray(cdd_ids),
                    "candidate_title": candidate_title_index,
                    "candidate_title_pad": candidate_attn_mask,
                    'his_id': np.asarray(his_ids),
                    "clicked_title": clicked_title_index,
                    "clicked_title_pad": clicked_attn_mask,
                    "his_mask": his_mask,
                    "labels": np.asarray([label])
                }

            else:
                candidate_title_pad = [(self.title_size - self.title_pad[impr_news][0])*[1] + self.title_pad[impr_news][0]*[0]]
                click_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]

                candidate_title_index = [self.news_title_array[impr_news]]
                clicked_title_index = self.news_title_array[his_ids]

                back_dic = {
                    "impression_index": impr_index,
                    "user_index": np.asarray(user_index),
                    'cdd_id': np.asarray(cdd_ids),
                    "candidate_title": np.asarray(candidate_title_index),
                    "candidate_title_pad": np.asarray(candidate_title_pad),
                    'his_id': np.asarray(his_ids),
                    "clicked_title": clicked_title_index,
                    "clicked_title_pad": np.asarray(click_title_pad),
                    "his_mask": his_mask,
                    "labels": np.asarray([label])
                }
            return back_dic

        else:
            raise ValueError


class MIND_test(IterableDataset):
    """ Iterator Dataset for MIND, yield every behaviors in each impression without labels

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
    """

    def __init__(self, hparams, news_file, behaviors_file):
        # initiate the whole iterator
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.k = hparams['k']

        # self.index2nid = hparams['news_id']

        self.vocab = getVocab(
            'data/dictionaries/vocab_{}.pkl'.format('_'.join(hparams['attrs'])))
        self.nid2index = getId2idx('data/dictionaries/nid2idx_large_test.json')
        self.uid2index = getId2idx('data/dictionaries/uid2idx_large.json')

        self.bert = False
        if 'bert' in hparams:
            self.bert = True
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['bert'])

        self.init_news()
        self.init_behaviors()

    def init_news(self):
        """ 
            init news information given news file, such as news_title_array.
        """

        title_token = [[0]*self.title_size]
        # category_token = [[0]]
        # subcategory_token = [[0]]

        title_pad = [[self.title_size]]

        titles = [['hello nlp']]

        with open(self.news_file, "r", encoding='utf-8') as rd:

            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(
                    self.col_spliter
                )

                titles.append([title])
                title = word_tokenize_vocab(title, self.vocab)

                title_token.append(
                    title[:self.title_size] + [0] * (self.title_size - len(title)))
                title_pad.append([max(self.title_size - len(title), 0)])

                # category_token.append([self.vocab[vert]])
                # subcategory_token.append([self.vocab[subvert]])

        self.titles = titles
        self.news_title_array = np.asarray(title_token)

        # self.news_category_array = np.asarray(category_token)
        # self.news_subcategory_array = np.asarray(subcategory_token)

        self.title_pad = np.asarray(title_pad)

    def init_behaviors(self):
        """ 
            init behavior logs given behaviors file.
        """

        # list of click history of each log
        self.histories = []
        # list of impression of each log
        self.imprs = []
        # list of impressions
        self.impr_indexes = []
        # user ids
        self.uindexes = []
        # history padding
        self.his_pad = []

        with open(self.behaviors_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                impr_index, uid, time, history, impr = idx.strip(
                    "\n").split(self.col_spliter)

                history = [self.nid2index[i] for i in history.split()]

                if self.k:
                    # guarantee there are at least k history not masked
                    self.his_pad.append(
                        min(max(self.his_size - len(history), 0), self.his_size - self.k))
                else:
                    self.his_pad.append(max(self.his_size - len(history), 0))

                # tailor user's history or pad 0
                history = history[:self.his_size] + \
                    [0] * (self.his_size - len(history))

                impr_news = [self.nid2index[i] for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.impr_indexes.append(int(impr_index))
                self.uindexes.append(uindex)

    def __iter__(self):
        """ parse behavior logs into training examples

        Yields:
            dict of training data, including 1 candidate news word vector, |his_size+1| clicked news word vector etc.
        """
        for impr_index in self.impr_indexes:
            index = impr_index - 1
            impr = self.imprs[index]

            for news in impr:
                # indicate the candidate news title vector from impression
                candidate_title_index = []
                # candidate_category_index = []
                # candidate_subcategory_index = []

                # indicate user ID
                user_index = []
                # indicate history mask
                his_mask = np.zeros((self.his_size, 1), dtype=bool)
                
                cdd_ids = [news]
                his_ids = self.histories[index]

                user_index.append(self.uindexes[index])

                # in case the user has no history records, do not mask
                if self.his_pad[index] == self.his_size or self.his_pad[index] == 0:
                    his_mask = his_mask
                else:
                    # print(self.his_pad[idx])
                    his_mask[-self.his_pad[index]:] = [True]

                if self.bert:
                    encoded_title = self.tokenizer(self.titles[news],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                    candidate_title_index = encoded_title['input_ids']
                    candidate_attn_mask = encoded_title['attention_mask']

                    encoded_title = self.tokenizer([self.titles[i] for i in self.histories[index]],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                    clicked_title_index = encoded_title['input_ids']
                    clicked_attn_mask = encoded_title['attention_mask']

                    back_dic = {
                        "impression_index":impr_index,
                        "user_index": np.asarray(user_index),
                        # "cdd_mask": np.asarray(neg_pad),
                        'cdd_id': np.asarray(cdd_ids),
                        "candidate_title": candidate_title_index,
                        # "candidate_category": candidate_category_index,
                        # "candidate_subcategory": candidate_subcategory_index,
                        "candidate_title_pad": candidate_attn_mask,
                        'his_id': np.asarray(his_ids),
                        "clicked_title": clicked_title_index,
                        # "clicked_category":click_category_index,
                        # "clicked_subcategory":click_subcategory_index,
                        "clicked_title_pad": clicked_attn_mask,
                        "his_mask": his_mask,
                    }

                else:
                    # indicate whether the news is clicked
                    # append the news title vector corresponding to news variable, in order to generate [news_title_vector]
                    candidate_title_index.append(self.news_title_array[news])
                    # candidate_category_index.append(self.news_category_array[news])
                    # candidate_subcategory_index.append(self.news_subcategory_array[news])

                    # append the news title vector corresponding to news variable
                    click_title_index = self.news_title_array[self.histories[index]]
                    # click_category_index = self.news_category_array[self.histories[index]]
                    # click_subcategory_index = self.news_subcategory_array[self.histories[index]]

                    candidate_title_pad = [
                        (self.title_size - self.title_pad[news][0])*[1] + self.title_pad[news][0]*[0]]
                    click_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0]
                                    for i in self.title_pad[self.histories[index]]]
                    
                    back_dic = {
                        "impression_index": impr_index,
                        "user_index": np.asarray(user_index),
                        'cdd_id': np.asarray(cdd_ids),
                        "candidate_title": np.asarray(candidate_title_index),
                        # "candidate_category": np.asarray(candidate_category_index),
                        # "candidate_subcategory": np.asarray(candidate_subcategory_index),
                        "candidate_title_pad": np.asarray(candidate_title_pad),
                        'his_id': np.asarray(his_ids),
                        "clicked_title": click_title_index,
                        # "clicked_category":click_category_index,
                        # "clicked_subcategory":click_subcategory_index,
                        "clicked_title_pad": np.asarray(click_title_pad),
                        "his_mask": his_mask
                    }

                yield back_dic


class MIND_news(Dataset):
    """ Map Dataset for MIND, return each news, intended for pipeline(obtaining news representation in advance)

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        mode(str): train/test
    """

    def __init__(self, hparams, news_file):
        # initiate the whole iterator
        self.npratio = hparams['npratio']
        self.news_file = news_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.attrs = hparams['attrs']
        self.k = hparams['k']

        mode = re.search(
            '{}_(.*)/'.format(hparams['scale']), news_file).group(1)

        self.vocab = getVocab(
            'data/dictionaries/vocab_{}.pkl'.format('_'.join(hparams['attrs'])))
        self.nid2index = getId2idx(
            'data/dictionaries/nid2idx_{}_{}.json'.format(hparams['scale'], mode))

    def __len__(self):
        if not hasattr(self, "news_title_array"):
            self.init_news()

        return len(self.news_title_array)

    def init_news(self):
        """ 
            init news information given news file, such as news_title_array.
        """

        title_token = []
        # category_token = [[0]]
        # subcategory_token = [[0]]

        title_pad = []
        news_ids = []

        with open(self.news_file, "r", encoding='utf-8') as rd:

            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(
                    self.col_spliter
                )

                title = word_tokenize_vocab(title, self.vocab)
                title_token.append(
                    title[:self.title_size] + [0] * (self.title_size - len(title)))
                title_pad.append([max(self.title_size - len(title), 0)])
                # category_token.append([self.vocab[vert]])
                # subcategory_token.append([self.vocab[subvert]])
                news_ids.append(self.nid2index[nid])

        self.news_title_array = np.asarray(title_token)
        # self.news_category_array = np.asarray(category_token)
        # self.news_subcategory_array = np.asarray(subcategory_token)

        self.title_pad = np.asarray(title_pad)
        self.news_ids = news_ids

    def __getitem__(self, idx):
        """ parse behavior log No.idx to training example

        Args:
            idx (int): impression index, start from zero

        Returns:
            dict of training data, including |npratio+1| candidate news word vector, |his_size+1| clicked news word vector etc.
        """
        if not hasattr(self, "news_title_array"):
            self.init_news()

        candidate_title_pad = [(self.title_size - self.title_pad[idx][0])*[1] + self.title_pad[idx][0]*[0]]
        return {
            "candidate_title": np.asarray([self.news_title_array[idx]]),
            "news_id": self.news_ids[idx],
            "candidate_title_pad":np.asarray(candidate_title_pad)
        }
