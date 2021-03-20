import numpy as np
import re
from transformers import AutoTokenizer
from torch.utils.data import Dataset, IterableDataset
from .utils import newsample, getId2idx, word_tokenize_vocab, getVocab


class MIND(IterableDataset):
    """ Iterable Dataset for MIND, yield positive samples in each impression if npratio is declared, yield every candidate news samples in each impression if not

    Args:
        hparams(dict): pre-defined dictionary of hyper parameters
        news_file(str): path of news_file
        behaviors_file(str): path of behaviors_file
        shuffle(bool): whether to shuffle the order of impressions
    """

    def __init__(self, hparams, news_file, behaviors_file, shuffle=False, npratio=None):
        # initiate the whole iterator
        self.npratio = hparams['npratio']
        if npratio is not None:
            self.npratio = npratio

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = '\t'
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.attrs = hparams['attrs']
        self.k = hparams['k']

        # always TRUE
        # self.index2nid = hparams['news_id']

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
        self.init_behaviors(shuffle=shuffle)

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

    def init_behaviors(self, shuffle):
        """ 
            init behavior logs given behaviors file.
        """

        # list of click history of each log
        self.histories = []
        # list of impression of each log
        self.imprs = []
        # list of labels of each news of each log
        self.labels = []
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

                impr_news = [self.nid2index[i.split(
                    "-")[0]] for i in impr.split()]

                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(int(impr_index))
                self.uindexes.append(uindex)

        if shuffle:
            s = np.arange(0, len(self.impr_indexes), 1)
            np.random.shuffle(s)
            self.histories = np.asarray(self.histories)[s].tolist()
            self.imprs = np.asarray(self.imprs, dtype=object)[s].tolist()
            self.labels = np.asarray(self.labels, dtype=object)[s].tolist()
            self.impr_indexes = np.asarray(self.impr_indexes)[s].tolist()
            self.uindexes = np.asarray(self.uindexes)[s].tolist()

    def __iter__(self):
        """ 
            yield dict of training data, including |npratio+1| candidate news word vector, |his_size+1| clicked news word vector etc.
        """
        for impr_index in self.impr_indexes:
            # impr_index starts from 1, all list related with behaviors start from 0
            index = impr_index - 1
            impr = self.imprs[index]
            impr_label = self.labels[index]

            if self.mode == 'train':
                # user clicked news
                poss = []
                # user not clicked news
                negs = []

                for news, click in zip(impr, impr_label):
                    if click == 1:
                        poss.append(news)
                    else:
                        negs.append(news)

                for p in poss:
                    # every click should be learnt

                    candidate_title_index = []
                    # candidate_category_index = []
                    # candidate_subcategory_index = []
                    his_mask = np.zeros((self.his_size, 1), dtype=bool)

                    label = [1] + [0] * self.npratio

                    neg_list, neg_pad = newsample(negs, self.npratio)

                    cdd_ids = [p]+neg_list
                    his_ids = self.histories[index]

                    # in case the user has no history records, do not mask
                    if self.his_pad[index] == self.his_size or self.his_pad[index] == 0:
                        his_mask = his_mask
                    else:
                        his_mask[-self.his_pad[index]:] = [True]

                    user_index = [self.uindexes[index]]

                    # pad in candidate
                    # candidate_mask = [1] * neg_pad + [0] * (self.npratio + 1 - neg_pad)

                    # impression_index = [self.impr_indexes[index]]
                    
                    if self.bert:
                        encoded_title = self.tokenizer([self.titles[i] for i in [p] + neg_list],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                        candidate_title_index = encoded_title['input_ids']
                        candidate_attn_mask = encoded_title['attention_mask']

                        encoded_title = self.tokenizer([self.titles[i] for i in self.histories[index]],padding='max_length',truncation=True,max_length=self.title_size,is_split_into_words=True,return_tensors='np',return_attention_mask=True)
                        clicked_title_index = encoded_title['input_ids']
                        clicked_attn_mask = encoded_title['attention_mask']

                        back_dic = {
                            # "impression_index":np.asarray(impression_index),
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
                            "labels": np.asarray(label)
                        }
                    else:
                        # pad in title
                        candidate_title_pad = [
                            (self.title_size - i[0])*[1] + i[0]*[0] for i in self.title_pad[[p] + neg_list]]
                        click_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0]
                                        for i in self.title_pad[self.histories[index]]]
                        
                        candidate_title_index = self.news_title_array[[p] + neg_list]
                        # candidate_category_index = self.news_category_array[[p] + neg_list]
                        # candidate_subcategory_index = self.news_subcategory_array[[p] + neg_list]

                        clicked_title_index = self.news_title_array[self.histories[index]]
                        # click_category_index = self.news_category_array[self.histories[index]]
                        # click_subcategory_index = self.news_subcategory_array[self.histories[index]]

                        back_dic = {
                            # "impression_index":np.asarray(impression_index),
                            "user_index": np.asarray(user_index),
                            # "cdd_mask": np.asarray(neg_pad),
                            'cdd_id': np.asarray(cdd_ids),
                            "candidate_title": candidate_title_index,
                            # "candidate_category": candidate_category_index,
                            # "candidate_subcategory": candidate_subcategory_index,
                            "candidate_title_pad": np.asarray(candidate_title_pad),
                            'his_id': np.asarray(his_ids),
                            "clicked_title": clicked_title_index,
                            # "clicked_category":click_category_index,
                            # "clicked_subcategory":click_subcategory_index,
                            "clicked_title_pad": np.asarray(click_title_pad),
                            "his_mask": his_mask,
                            "labels": np.asarray(label)
                        }

                    yield back_dic
            elif self.mode == 'dev':
                for news, label in zip(impr, impr_label):
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
                            "labels": np.asarray([label])
                        }

                    else:
                        candidate_title_pad = [
                            (self.title_size - self.title_pad[news][0])*[1] + self.title_pad[news][0]*[0]]
                        click_title_pad = [(self.title_size - i[0])*[1] + i[0]*[0]
                                        for i in self.title_pad[self.histories[index]]]

                        # append the news title vector corresponding to news variable, in order to generate [news_title_vector]
                        candidate_title_index.append(self.news_title_array[news])
                        # candidate_category_index.append(self.news_category_array[news])
                        # candidate_subcategory_index.append(self.news_subcategory_array[news])

                        # append the news title vector corresponding to news variable
                        clicked_title_index = self.news_title_array[self.histories[index]]
                        # click_category_index = self.news_category_array[self.histories[index]]
                        # click_subcategory_index = self.news_subcategory_array[self.histories[index]]

                        back_dic = {
                            "impression_index": impr_index,
                            "user_index": np.asarray(user_index),
                            # "cdd_mask": np.asarray(neg_pad),
                            'cdd_id': np.asarray(cdd_ids),
                            "candidate_title": candidate_title_index,
                            # "candidate_category": candidate_category_index,
                            # "candidate_subcategory": candidate_subcategory_index,
                            "candidate_title_pad": np.asarray(candidate_title_pad),
                            'his_id': np.asarray(his_ids),
                            "clicked_title": clicked_title_index,
                            # "clicked_category":click_category_index,
                            # "clicked_subcategory":click_subcategory_index,
                            "clicked_title_pad": np.asarray(click_title_pad),
                            "his_mask": his_mask,
                            "labels": np.asarray([label])
                        }

                    yield back_dic

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
