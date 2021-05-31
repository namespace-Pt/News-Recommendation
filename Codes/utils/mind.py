import numpy as np
import re
from torch.utils.data import Dataset
from utils.utils import newsample, getId2idx, tokenize, getVocab

class MIND(Dataset):
    """ Map Style Dataset for MIND, return positive samples with negative sampling when training, or return each sample when developing.

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, hparams, news_file, behaviors_file, shuffle_pos=False, validate=False):
        # initiate the whole iterator
        self.npratio = hparams['npratio']
        self.shuffle_pos = shuffle_pos

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.abs_size = hparams['abs_size']
        self.his_size = hparams['his_size']

        self.onehot = hparams['onehot']
        self.k = hparams['k']
        self.mode = re.search(
            'MIND/.*_(.*)/news', news_file).group(1)

        # there are only two types of vocabulary
        self.vocab = getVocab('data/dictionaries/vocab_whole.pkl')

        self.nid2index = getId2idx(
            'data/dictionaries/nid2idx_{}_{}.json'.format(hparams['scale'], self.mode))
        self.uid2index = getId2idx(
            'data/dictionaries/uid2idx_{}.json'.format(hparams['scale']))
        self.vert2onehot = getId2idx(
            'data/dictionaries/vert2onehot.json'
        )
        self.subvert2onehot = getId2idx(
            'data/dictionaries/subvert2onehot.json'
        )

        if validate:
            self.mode = 'dev'

        self.init_news()
        self.init_behaviors()

    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        titles = [[1]*self.title_size]
        title_pad = [[self.title_size]]

        with open(self.news_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(self.col_spliter)

                title_token = tokenize(title, self.vocab)
                titles.append(title_token[:self.title_size] + [1] * (self.title_size - len(title_token)))
                title_pad.append([max(self.title_size - len(title_token), 0)])

        # self.titles = titles
        self.news_title_array = np.asarray(titles)
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

        # store every behaviors
        elif self.mode == 'test':
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
                    impr_news = [self.nid2index[i] for i in impr.split()]
                    # user will always in uid2index
                    uindex = self.uid2index[uid]

                    # store every impression
                    for news in impr_news:
                        self.imprs.append((impr_index, news))

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

            cdd_ids = [impr_news] + neg_list
            label = [1] + [0] * self.npratio

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

            # pad in title
            candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
            clicked_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]

            candidate_title_index = self.news_title_array[cdd_ids]
            clicked_title_index = self.news_title_array[his_ids]
            back_dic = {
                "user_index": np.asarray(user_index),
                # "cdd_mask": np.asarray(neg_pad),
                'cdd_id': np.asarray(cdd_ids),
                "candidate_title": candidate_title_index,
                "candidate_title_pad": np.asarray(candidate_title_pad),
                'his_id': np.asarray(his_ids),
                "clicked_title": clicked_title_index,
                "clicked_title_pad": np.asarray(clicked_title_pad),
                "his_mask": his_mask,
                "labels": label
            }

            return back_dic

        # each time called return one sample, and no labels
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

            candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
            clicked_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]
            candidate_title_index = [self.news_title_array[impr_news]]
            clicked_title_index = self.news_title_array[his_ids]
            back_dic = {
                "impression_index": impr_index + 1,
                "user_index": np.asarray(user_index),
                'cdd_id': np.asarray(cdd_ids),
                "candidate_title": np.asarray(candidate_title_index),
                "candidate_title_pad": np.asarray(candidate_title_pad),
                'his_id': np.asarray(his_ids),
                "clicked_title": clicked_title_index,
                "clicked_title_pad": np.asarray(clicked_title_pad),
                "his_mask": his_mask,
                "labels": np.asarray([label])
            }

            return back_dic

        elif self.mode == 'test':
            cdd_ids = [impr_news]

            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1), dtype=bool)
            his_ids = self.histories[impr_index]

            user_index = [self.uindexes[impr_index]]

            # in case the user has no history records, do not mask
            if self.his_pad[impr_index] == self.his_size or self.his_pad[impr_index] == 0:
                his_mask = his_mask
            else:
                his_mask[-self.his_pad[impr_index]:] = [True]

            candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
            clicked_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]

            candidate_title_index = [self.news_title_array[impr_news]]
            clicked_title_index = self.news_title_array[his_ids]
            back_dic = {
                "impression_index": impr_index + 1,
                "user_index": np.asarray(user_index),
                'cdd_id': np.asarray(cdd_ids),
                "candidate_title": np.asarray(candidate_title_index),
                "candidate_title_pad": np.asarray(candidate_title_pad),
                'his_id': np.asarray(his_ids),
                "clicked_title": clicked_title_index,
                "clicked_title_pad": np.asarray(clicked_title_pad),
                "his_mask": his_mask
            }

            return back_dic

        else:
            raise ValueError("Mode {} not defined".format(self.mode))


class MIND_all(Dataset):
    """ Map Style Dataset for MIND, return positive samples with negative sampling when training, or return each sample when developing.

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, hparams, news_file, behaviors_file, shuffle_pos=False, validate=False):
        # initiate the whole iterator
        self.npratio = hparams['npratio']
        self.shuffle_pos = shuffle_pos

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.abs_size = hparams['abs_size']
        self.his_size = hparams['his_size']

        self.onehot = hparams['onehot']
        self.k = hparams['k']
        self.mode = re.search(
            'MIND/.*_(.*)/news', news_file).group(1)

        # there are only two types of vocabulary
        self.vocab = getVocab('data/dictionaries/vocab_whole.pkl')

        self.nid2index = getId2idx(
            'data/dictionaries/nid2idx_{}_{}.json'.format(hparams['scale'], self.mode))
        self.uid2index = getId2idx(
            'data/dictionaries/uid2idx_{}.json'.format(hparams['scale']))
        self.vert2onehot = getId2idx(
            'data/dictionaries/vert2onehot.json'
        )
        self.subvert2onehot = getId2idx(
            'data/dictionaries/subvert2onehot.json'
        )

        if validate:
            self.mode = 'dev'

        self.init_news()
        self.init_behaviors()

    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        titles = [[1]*self.title_size]
        title_pad = [[self.title_size]]
        abstracts = [[1]*self.abs_size]
        abs_pad = [[self.abs_size]]

        # pure text of the title
        # titles = [['hello MIND']]
        categories = [[1]]
        subcategories = [[1]]

        with open(self.news_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(self.col_spliter)

                title_token = tokenize(title, self.vocab)
                titles.append(title_token[:self.title_size] + [1] * (self.title_size - len(title_token)))
                title_pad.append([max(self.title_size - len(title_token), 0)])

                abs_token = tokenize(ab, self.vocab)
                abstracts.append(abs_token[:self.abs_size] + [1] * (self.abs_size - len(abs_token)))
                abs_pad.append([max(self.abs_size - len(abs_token), 0)])

                categories.append(tokenize(vert, self.vocab))
                subcategories.append(tokenize(subvert, self.vocab))

        # self.titles = titles
        self.news_title_array = np.asarray(titles)
        self.title_pad = np.asarray(title_pad)
        self.abs_array = np.asarray(abstracts)
        self.abs_pad = np.asarray(abs_pad)
        self.vert_array = np.asarray(categories)
        self.subvert_array = np.asarray(subcategories)

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

        # store every behaviors
        elif self.mode == 'test':
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
                    impr_news = [self.nid2index[i] for i in impr.split()]
                    # user will always in uid2index
                    uindex = self.uid2index[uid]

                    # store every impression
                    for news in impr_news:
                        self.imprs.append((impr_index, news))

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

            cdd_ids = [impr_news] + neg_list
            label = [1] + [0] * self.npratio

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

            # pad in title
            candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
            clicked_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]
            candidate_abs_pad = [(self.abs_size - i[0])*[1] + i[0]*[0] for i in self.abs_pad[cdd_ids]]
            clicked_abs_pad = [(self.abs_size - i[0])*[1] + i[0]*[0] for i in self.abs_pad[his_ids]]

            candidate_title_index = self.news_title_array[cdd_ids]
            clicked_title_index = self.news_title_array[his_ids]
            candidate_abs_index = self.abs_array[cdd_ids]
            clicked_abs_index = self.abs_array[his_ids]
            candidate_vert_index = self.vert_array[cdd_ids]
            clicked_vert_index = self.vert_array[his_ids]
            candidate_subvert_index = self.subvert_array[cdd_ids]
            clicked_subvert_index = self.subvert_array[his_ids]

            back_dic = {
                "user_index": np.asarray(user_index),
                # "cdd_mask": np.asarray(neg_pad),
                'cdd_id': np.asarray(cdd_ids),
                "candidate_title": candidate_title_index,
                "candidate_title_pad": np.asarray(candidate_title_pad),
                "candidate_abs": candidate_abs_index,
                "candidate_abs_pad": np.asarray(candidate_abs_pad),
                "candidate_vert": candidate_vert_index,
                "candidate_subvert": candidate_subvert_index,
                'his_id': np.asarray(his_ids),
                "clicked_title": clicked_title_index,
                "clicked_title_pad": np.asarray(clicked_title_pad),
                "clicked_abs": clicked_abs_index,
                "clicked_abs_pad": np.asarray(clicked_abs_pad),
                "clicked_vert": clicked_vert_index,
                "clicked_subvert": clicked_subvert_index,
                "his_mask": his_mask,
                "labels": label
            }

            if self.onehot:
                candidate_vert_onehot = [self.vert2onehot[str(i[0])] for i in candidate_vert_index]
                clicked_vert_onehot = [self.vert2onehot[str(i[0])] for i in clicked_vert_index]
                candidate_subvert_onehot = [self.subvert2onehot[str(i[0])] for i in candidate_subvert_index]
                clicked_subvert_onehot = [self.subvert2onehot[str(i[0])] for i in clicked_subvert_index]

                back_dic['candidate_vert_onehot'] = np.asarray(candidate_vert_onehot)
                back_dic['clicked_vert_onehot'] = np.asarray(clicked_vert_onehot)
                back_dic['candidate_subvert_onehot'] = np.asarray(candidate_subvert_onehot)
                back_dic['clicked_subvert_onehot'] = np.asarray(clicked_subvert_onehot)

            return back_dic

        # each time called return one sample, and no labels
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

            candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
            clicked_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]
            candidate_abs_pad = [(self.abs_size - i[0])*[1] + i[0]*[0] for i in self.abs_pad[cdd_ids]]
            clicked_abs_pad = [(self.abs_size - i[0])*[1] + i[0]*[0] for i in self.abs_pad[his_ids]]

            candidate_title_index = [self.news_title_array[impr_news]]
            clicked_title_index = self.news_title_array[his_ids]
            candidate_abs_index = self.abs_array[cdd_ids]
            clicked_abs_index = self.abs_array[his_ids]
            candidate_vert_index = self.vert_array[cdd_ids]
            clicked_vert_index = self.vert_array[his_ids]
            candidate_subvert_index = self.subvert_array[cdd_ids]
            clicked_subvert_index = self.subvert_array[his_ids]

            back_dic = {
                "impression_index": impr_index + 1,
                "user_index": np.asarray(user_index),
                'cdd_id': np.asarray(cdd_ids),
                "candidate_title": np.asarray(candidate_title_index),
                "candidate_title_pad": np.asarray(candidate_title_pad),
                "candidate_abs": candidate_abs_index,
                "candidate_abs_pad": np.asarray(candidate_abs_pad),
                "candidate_vert": candidate_vert_index,
                "candidate_subvert": candidate_subvert_index,
                'his_id': np.asarray(his_ids),
                "clicked_title": clicked_title_index,
                "clicked_title_pad": np.asarray(clicked_title_pad),
                "clicked_abs": clicked_abs_index,
                "clicked_abs_pad": np.asarray(clicked_abs_pad),
                "clicked_vert": clicked_vert_index,
                "clicked_subvert": clicked_subvert_index,
                "his_mask": his_mask,
                "labels": np.asarray([label])
            }

            if self.onehot:
                candidate_vert_onehot = [self.vert2onehot[str(i[0])] for i in candidate_vert_index]
                clicked_vert_onehot = [self.vert2onehot[str(i[0])] for i in clicked_vert_index]

                candidate_subvert_onehot = [self.subvert2onehot[str(i[0])] for i in candidate_subvert_index]
                clicked_subvert_onehot = [self.subvert2onehot[str(i[0])] for i in clicked_subvert_index]

                back_dic['candidate_vert_onehot'] = np.asarray(candidate_vert_onehot)
                back_dic['clicked_vert_onehot'] = np.asarray(clicked_vert_onehot)
                back_dic['candidate_subvert_onehot'] = np.asarray(candidate_subvert_onehot)
                back_dic['clicked_subvert_onehot'] = np.asarray(clicked_subvert_onehot)

            return back_dic

        elif self.mode == 'test':
            cdd_ids = [impr_news]

            # true means the corresponding history news is padded
            his_mask = np.zeros((self.his_size, 1), dtype=bool)
            his_ids = self.histories[impr_index]

            user_index = [self.uindexes[impr_index]]

            # in case the user has no history records, do not mask
            if self.his_pad[impr_index] == self.his_size or self.his_pad[impr_index] == 0:
                his_mask = his_mask
            else:
                his_mask[-self.his_pad[impr_index]:] = [True]

            candidate_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[cdd_ids]]
            clicked_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[his_ids]]
            candidate_abs_pad = [(self.abs_size - i[0])*[1] + i[0]*[0] for i in self.abs_pad[cdd_ids]]
            clicked_abs_pad = [(self.abs_size - i[0])*[1] + i[0]*[0] for i in self.abs_pad[his_ids]]

            candidate_title_index = [self.news_title_array[impr_news]]
            clicked_title_index = self.news_title_array[his_ids]
            candidate_abs_index = self.abs_array[cdd_ids]
            clicked_abs_index = self.abs_array[his_ids]
            candidate_vert_index = self.vert_array[cdd_ids]
            clicked_vert_index = self.vert_array[his_ids]
            candidate_subvert_index = self.subvert_array[cdd_ids]
            clicked_subvert_index = self.subvert_array[his_ids]

            back_dic = {
                "impression_index": impr_index + 1,
                "user_index": np.asarray(user_index),
                'cdd_id': np.asarray(cdd_ids),
                "candidate_title": np.asarray(candidate_title_index),
                "candidate_title_pad": np.asarray(candidate_title_pad),
                "candidate_abs": candidate_abs_index,
                "candidate_abs_pad": np.asarray(candidate_abs_pad),
                "candidate_vert": candidate_vert_index,
                "candidate_subvert": candidate_subvert_index,
                'his_id': np.asarray(his_ids),
                "clicked_title": clicked_title_index,
                "clicked_title_pad": np.asarray(clicked_title_pad),
                "clicked_abs": clicked_abs_index,
                "clicked_abs_pad": np.asarray(clicked_abs_pad),
                "clicked_vert": clicked_vert_index,
                "clicked_subvert": clicked_subvert_index,
                "his_mask": his_mask
            }

            if self.onehot:
                candidate_vert_onehot = [self.vert2onehot[str(i[0])] for i in candidate_vert_index]
                clicked_vert_onehot = [self.vert2onehot[str(i[0])] for i in clicked_vert_index]

                candidate_subvert_onehot = [self.subvert2onehot[str(i[0])] for i in candidate_subvert_index]
                clicked_subvert_onehot = [self.subvert2onehot[str(i[0])] for i in clicked_subvert_index]

                back_dic['candidate_vert_onehot'] = np.asarray(candidate_vert_onehot)
                back_dic['clicked_vert_onehot'] = np.asarray(clicked_vert_onehot)
                back_dic['candidate_subvert_onehot'] = np.asarray(candidate_subvert_onehot)
                back_dic['clicked_subvert_onehot'] = np.asarray(clicked_subvert_onehot)
            return back_dic

        else:
            raise ValueError("Mode {} not defined".format(self.mode))


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
        # self.attrs = hparams['attrs']

        mode = re.search(
            '{}_(.*)/'.format(hparams['scale']), news_file).group(1)

        self.vocab = getVocab(
            'data/dictionaries/vocab_whole.pkl')
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
        title_pad = []
        news_ids = []

        with open(self.news_file, "r", encoding='utf-8') as rd:

            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(
                    self.col_spliter
                )

                title = tokenize(title, self.vocab)
                title_token.append(
                    title[:self.title_size] + [0] * (self.title_size - len(title)))
                title_pad.append([max(self.title_size - len(title), 0)])
                news_ids.append(self.nid2index[nid])

        self.news_title_array = np.asarray(title_token)

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
            "cdd_id": self.news_ids[idx],
            "candidate_title_pad":np.asarray(candidate_title_pad)
        }


class MIND_impr(Dataset):
    """ Map Style Dataset for MIND, return each impression once

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, hparams, news_file, behaviors_file, shuffle_pos=False, validate=False):
        # initiate the whole iterator
        self.npratio = hparams['npratio']
        self.shuffle_pos = shuffle_pos

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.abs_size = hparams['abs_size']
        self.his_size = hparams['his_size']

        self.onehot = hparams['onehot']
        self.k = hparams['k']

        # there are only two types of vocabulary
        self.vocab = getVocab('data/dictionaries/vocab_whole.pkl')

        self.nid2index = getId2idx(
            'data/dictionaries/nid2idx_{}_{}.json'.format(hparams['scale'], 'dev'))
        self.uid2index = getId2idx(
            'data/dictionaries/uid2idx_{}.json'.format(hparams['scale']))
        self.vert2onehot = getId2idx(
            'data/dictionaries/vert2onehot.json'
        )
        self.subvert2onehot = getId2idx(
            'data/dictionaries/subvert2onehot.json'
        )

        self.mode = 'dev'

        self.init_news()
        self.init_behaviors()

    def init_news(self):
        """
            init news information given news file, such as news_title_array.
        """

        # VERY IMPORTANT!!! FIXME
        # The nid2idx dictionary must follow the original order of news in news.tsv

        titles = [[1]*self.title_size]
        title_pad = [[self.title_size]]

        with open(self.news_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(self.col_spliter)

                title_token = tokenize(title, self.vocab)
                titles.append(title_token[:self.title_size] + [1] * (self.title_size - len(title_token)))
                title_pad.append([max(self.title_size - len(title_token), 0)])

        # self.titles = titles
        self.news_title_array = np.asarray(titles)
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

        # list of every candidate news index along with its impression index and label
        self.imprs = []

        with open(self.behaviors_file, "r", encoding='utf-8') as rd:
            for idx in rd:
                impr_index, uid, time, history, impr = idx.strip("\n").split(self.col_spliter)
                impr_index = int(impr_index) - 1

                history = [self.nid2index[i] for i in history.split()]
                # tailor user's history or pad 0
                history = history[:self.his_size] + [0] * (self.his_size - len(history))
                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                labels = [int(i.split("-")[1]) for i in impr.split()]
                # user will always in uid2index
                uindex = self.uid2index[uid]

                # store every impression
                self.imprs.append((impr_index, impr_news[0], labels[0]))

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

        cdd_ids = [impr_news]

        his_ids = self.histories[impr_index]

        user_index = [self.uindexes[impr_index]]
        label = impr[2]

        candidate_title_index = [self.news_title_array[impr_news]]
        clicked_title_index = self.news_title_array[his_ids]
        back_dic = {
            "impression_index": impr_index + 1,
            "user_index": np.asarray(user_index),
            'cdd_id': np.asarray(cdd_ids),
            "candidate_title": np.asarray(candidate_title_index),
            'his_id': np.asarray(his_ids),
            "clicked_title": clicked_title_index,
            "labels": np.asarray([label])
        }

        return back_dic