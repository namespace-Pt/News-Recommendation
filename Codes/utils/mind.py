'''
Author: Pt
Date: 2020-11-19 21:58:13
LastEditTime: 2020-11-20 10:41:37
Description: MIND dataset
'''
import torch
import numpy as np
from torch.utils.data import Dataset,IterableDataset
from .utils import newsample,getId2idx,word_tokenize_vocab,getVocab,constructBasicDict

class MIND_map(Dataset):
    """ batch iterator for MIND dataset

        Args:
        hparams: pre-defined dictionary of hyper parameters
    """
    def __init__(self,hparams,mode,npratio,news_file,behaviors_file,col_spliter='\t'):
        # initiate the whole iterator
        self.npratio = npratio
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = col_spliter        
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device('cpu')

        self.vocab = getVocab('data/vocab_{}_{}_{}.pkl'.format(hparams['mode'],mode,'_'.join(hparams['attrs'])))
        self.nid2index = getId2idx('data/nid2idx_{}_{}.json'.format(hparams['mode'],mode))
        self.uid2index = getId2idx('data/uid2idx_{}_{}.json'.format(hparams['mode'],mode))
    
    def __len__(self):
        if not hasattr(self, "news_title_array"):
            self.init_news()

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors()

        return len(self.labels)
    
    def init_news(self):
        """ 
            init news information given news file, such as news_title_array.
        """

        title_token = [[0]*self.title_size]
        category_token = [[0]]
        subcategory_token = [[0]]
        
        with open(self.news_file,"r",encoding='utf-8') as rd:

            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(
                    self.col_spliter
                )

                title = word_tokenize_vocab(title,self.vocab)
                title_token.append(title[:self.title_size] + [0] * (self.title_size - len(title)))
                category_token.append([self.vocab[vert]])
                subcategory_token.append([self.vocab[subvert]])
        
        self.news_title_array = np.asarray(title_token)
        self.news_category_array = np.asarray(category_token)
        self.news_subcategory_array = np.asarray(subcategory_token)

    def init_behaviors(self):
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

        with open(self.behaviors_file, "r",encoding='utf-8') as rd:
            impr_index = 0
            for idx in rd:
                uid, time, history, impr = idx.strip("\n").split(self.col_spliter)[-4:]
                
                history = [self.nid2index[i] for i in history.split()]
                
                # tailor user's history or pad 0
                history = history[:self.his_size] + [0] * (self.his_size - len(history))
        
                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                
                label = [int(i.split("-")[1]) for i in impr.split()]
                
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def __getitem__(self, idx):
        """Parse one behavior sample into |candidates| feature values, each of which consists of
        one single candidate title vector when npratio < 0 or npratio+1 candidate title vectors when npratio > 0

        if npratio is larger than 0, return negtive sampled result.

        npratio is for negtive sampling (used in softmax)
        
        Args:
            idx (int): sample index/impression index

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_title_index, clicked_title_index.
        """
        if not hasattr(self, "news_title_array"):
            self.init_news()

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors()
            

        impr_label = self.labels[idx]
        impr = self.imprs[idx]

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
            candidate_title_index = []
            candidate_category_index = []
            candidate_subcategory_index = []
            user_index = []
            
            label = [1] + [0] * self.npratio

            neg_list = newsample(negs, self.npratio)

            candidate_title_index = self.news_title_array[[p] + neg_list]
            candidate_category_index = self.news_category_array[[p] + neg_list]
            candidate_subcategory_index = self.news_subcategory_array[[p] + neg_list]
            
            click_title_index = self.news_title_array[self.histories[idx]]
            click_category_index = self.news_category_array[self.histories[idx]]
            click_subcategory_index = self.news_subcategory_array[self.histories[idx]]

            # impression_index not needed in training
            # impr_index = self.impr_indexes[idx]
            
            user_index.append(self.uindexes[idx])

            return {
                "user_index": np.asarray(user_index),
                "clicked_title": click_title_index,
                "clicked_category":click_category_index,
                "clicked_subcategory":click_subcategory_index,
                "candidate_title": candidate_title_index,
                "candidate_category": candidate_category_index,
                "candidate_subcategory": candidate_subcategory_index,
                # important to transfer to array, otherwise produces error in converted tensor
                # FIXME understand why
                "labels": np.asarray(label)
            }

class MIND_iter(IterableDataset):
    """ batch iterator for MIND dataset

        Args:
        hparams: pre-defined dictionary of hyper parameters
    """
    def __init__(self,hparams,mode,news_file,behaviors_file,col_spliter='\t'):
        # initiate the whole iterator
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = col_spliter        
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device('cpu')

        self.vocab = getVocab('data/vocab_{}_{}_{}.pkl'.format(hparams['mode'],mode,'_'.join(hparams['attrs'])))
        self.nid2index = getId2idx('data/nid2idx_{}_{}.json'.format(hparams['mode'],mode))
        self.uid2index = getId2idx('data/uid2idx_{}_{}.json'.format(hparams['mode'],mode))
    
    def init_news(self):
        """ 
            init news information given news file, such as news_title_array.
        """

        title_token = [[0]*self.title_size]
        category_token = [[0]]
        subcategory_token = [[0]]
        
        with open(self.news_file,"r",encoding='utf-8') as rd:

            for idx in rd:
                nid, vert, subvert, title, ab, url, _, _ = idx.strip("\n").split(
                    self.col_spliter
                )

                title = word_tokenize_vocab(title,self.vocab)
                title_token.append(title[:self.title_size] + [0] * (self.title_size - len(title)))
                category_token.append([self.vocab[vert]])
                subcategory_token.append([self.vocab[subvert]])
        
        self.news_title_array = np.asarray(title_token)
        self.news_category_array = np.asarray(category_token)
        self.news_subcategory_array = np.asarray(subcategory_token)

    def init_behaviors(self):
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

        with open(self.behaviors_file, "r",encoding='utf-8') as rd:
            impr_index = 0
            for idx in rd:
                uid, time, history, impr = idx.strip("\n").split(self.col_spliter)[-4:]
                
                history = [self.nid2index[i] for i in history.split()]
                
                # tailor user's history or pad 0
                history = history[:self.his_size] + [0] * (self.his_size - len(history))
        
                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                try:
                    label = [int(i.split("-")[1]) for i in impr.split()]
                except:
                    print(impr,impr_index)
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def __iter__(self):
        """Parse one behavior sample into |candidates| feature values, each of which consists of
        one single candidate title vector when npratio < 0 or npratio+1 candidate title vectors when npratio > 0

        if npratio is larger than 0, return negtive sampled result.

        npratio is for negtive sampling (used in softmax)
        
        Args:
            idx (int): sample index/impression index

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_title_index, clicked_title_index.
        """
        if not hasattr(self, "news_title_array"):
            self.init_news()

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors()
        
        indexes = np.arange(len(self.labels))

        for index,impr_label in enumerate(self.labels):
            impr = self.imprs[index]
        
            for news, label in zip(impr, impr_label):
                # indicate the candidate news title vector from impression
                candidate_title_index = []
                candidate_category_index = []
                candidate_subcategory_index = []

                # indicate the impression where the news in
                impr_index = []
                # indicate user ID
                user_index = []
                # indicate whether the news is clicked
                label = [label]
                # append the news title vector corresponding to news variable, in order to generate [news_title_vector]
                candidate_title_index.append(self.news_title_array[news])
                candidate_category_index.append(self.news_category_array[news])
                candidate_subcategory_index.append(self.news_subcategory_array[news])
                
                # append the news title vector corresponding to news variable
                click_title_index = self.news_title_array[self.histories[index]]
                click_category_index = self.news_category_array[self.histories[index]]
                click_subcategory_index = self.news_subcategory_array[self.histories[index]]

                impr_index = self.impr_indexes[index]
                user_index.append(self.uindexes[index])
                
                yield {
                    "impression_index": impr_index,
                    "user_index": np.asarray(user_index),
                    "clicked_title": click_title_index,
                    "clicked_category":click_category_index,
                    "clicked_subcategory":click_subcategory_index,
                    
                    # similarly, important to convert to numpy array rather than retaining list
                    "candidate_title": np.asarray(candidate_title_index),
                    "candidate_category": np.asarray(candidate_category_index),
                    "candidate_subcategory": np.asarray(candidate_subcategory_index),
                    "labels": np.asarray(label)
                }
    