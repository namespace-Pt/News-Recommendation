'''
Author: Pt
Date: 2020-11-13 19:39:12
LastEditTime: 2020-11-14 10:43:37
'''
import torch
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import numericalize_tokens_from_iterator
from .utils import newsample,getId2idx,word_tokenize_vocab,getVocab,constructBasicDict,news_token_generator_group

class MINDDataset(Dataset):
    def __init__(self,hparams,news_file,behaviors_file):
        # initiate the whole iterator
        self.mode = hparams['load_mode']
        self.col_spliter = hparams['col_spliter']
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.npratio = hparams['npratio']
        self.device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        
        self.vocab = getVocab('data/vocab_'+hparams['mode']+'.pkl')
        self.nid2index = getId2idx('data/nid2idx_'+hparams['mode']+'.json')
        self.uid2index = getId2idx('data/uid2idx_'+hparams['mode']+'.json')
    

    def _init_news(self):
        """ 
            init news, must be in order
        """

        title_token = [[0]*self.title_size]
        category_token = [[0]]
        subcategory_token = [[0]]
        
        with open(self.news_file,"r",encoding='utf-8') as rd:

            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                title = word_tokenize_vocab(title,self.vocab)
                title_token.append(title[:self.title_size] + [0] * (self.title_size - len(title)))
                category_token.append(self.vocab[vert])
                subcategory_token.append(self.vocab[subvert])


        "this version is fucking slower, iterator is slower than loop?"
        # tokenizer = word_tokenize
        # news_token_iterator = news_token_generator_group(self.news_file,tokenizer,self.vocab,self.mode)

        # for news_token in news_token_iterator:
            
        #     title_token.append(news_token[0][:self.title_size] + [0] * (self.title_size - len(news_token[0])))
        #     category_token.append(news_token[1])
        #     subcategory_token.append(news_token[2])
        
        self.news_title_array = np.asarray(title_token)
        self.news_category_array = np.asarray(category_token)
        self.news_subcategory_array = np.asarray(subcategory_token)
    
    def init_behaviors(self):
        """ init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """

        # list(newsID of history click)
        self.histories = []
        # list(newsID of impression news)
        self.imprs = []
        # list(click label of impression news)
        self.labels = []
        # index of impression
        self.impr_indexes = []
        # user ids
        self.uindexes = []

        with open(self.behaviors_file, "r",encoding='utf-8') as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                
                # tailor user's history or pad 0
                history = [0] * (self.his_size - len(history)) + history[
                    : self.his_size
                ]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def __len__(self):
        if not hasattr(self, "news_title_array"):
            self.init_news()

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors()
        
        return len(self.news_title_array)
        

    def __getitem__(self,idx):
        """Parse one behavior sample into |candidates| feature values, each of which consists of
        one single candidate title vector when npratio < 0 or npratio+1 candidate title vectors when npratio > 0

        if npratio is larger than 0, return negtive sampled result.

        npratio is for negtive sampling (used in softmax)
        
        Args:
            line (int): sample index/impression index

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_title_index, clicked_title_index.
        """
        if self.npratio > 0:
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
                
                impr_index = []
                user_index = []
                label = [1] + [0] * self.npratio

                neg_list = newsample(negs, self.npratio)

                candidate_title_index = self.news_title_array[[p] + neg_list]
                candidate_category_index = self.news_category_array[[p] + neg_list]
                candidate_subcategory_index = self.news_subcategory_array[[p] + neg_list]
                
                click_title_index = self.news_title_array[self.histories[line]]
                click_category_index = self.news_category_array[self.histories[line]]
                click_subcategory_index = self.news_subcategory_array[self.histories[line]]

                impr_index = self.impr_indexes[line]
                user_index.append(self.uindexes[line])


                """ haven't try to return multiple tensors """
                return (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    
                    click_title_index,
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                # indicate the candidate news title vector from impression
                candidate_title_index = []
                # indicate the impression where the news in
                impr_index = []
                # indicate user ID
                user_index = []
                # indicate whether the news is clicked
                label = [label]
                # append the news title vector corresponding to news variable, in order to generate [news_title_vector]
                candidate_title_index.append(self.news_title_array[news])
                # append the news title vector corresponding to news variable
                click_title_index = self.news_title_array[self.histories[line]]
                impr_index = self.impr_indexes[line]
                user_index.append(self.uindexes[line])
                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                )

    def load_data_from_file(self):
        """Read and parse data from news file and behavior file, generate batch_size of training examples, each of which contains
        an impression id, a user id, a union tensor of history clicked news' title tensor, a candidate news' title vector, a click label
        
        Returns:
            obj: An iterator that will yields batch of parsed results, in the format of dict.
            
        """

        if not hasattr(self, "news_title_array"):
            self.init_news()

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors()

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_title_indexes = []
        click_title_indexes = []
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                label,
                imp_index,
                user_index,
                candidate_title_index,
                click_title_index,
            ) in self.parser_one_line(index):

                # append one log in the batch
                candidate_title_indexes.append(candidate_title_index)
                click_title_indexes.append(click_title_index)
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)
                label_list.append(label)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_title_indexes,
                        click_title_indexes,
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    candidate_title_indexes = []
                    click_title_indexes = []
                    cnt = 0

        "in case the last few examples can't fill a batch, abandon them"

        # if cnt > 0:
        #     yield self._convert_data(
        #         label_list,
        #         imp_indexes,
        #         user_indexes,
        #         candidate_title_indexes,
        #         click_title_indexes,
        #     )

    def _convert_data(self,label_list,imp_indexes,user_indexes,candidate_title_indexes,click_title_indexes):
        """Convert data of one candidate into torch.tensor that are good for further model operation, 
        
        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.
            
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        labels = torch.tensor(label_list, dtype=torch.float32,device=self.device)
        # imp_indexes = torch.tensor(imp_indexes, dtype=torch.int64,device=self.device)
        user_indexes = torch.tensor(user_indexes, dtype=torch.int64,device=self.device)

        candidate_title_index_batch = torch.tensor(
            candidate_title_indexes, dtype=torch.int64,device=self.device
        )
        click_title_index_batch = torch.tensor(click_title_indexes, dtype=torch.int64,device=self.device)
        
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "labels": labels,
        }