import torch
import numpy as np
import os
from .utils import newsample,getId2idx,word_tokenize_vocab,getVocab,constructBasicDict

class MINDIterator():
    """ batch iterator for MIND dataset

        Args:
        hparams: pre-defined dictionary of hyper parameters
    """
    def __init__(self, hparams,mode,news_file,behaviors_file,col_spliter='\t'):
        # initiate the whole iterator
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = col_spliter        
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.npratio = hparams['npratio']
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

            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                title = word_tokenize_vocab(title,self.vocab)
                title_token.append(title[:self.title_size] + [0] * (self.title_size - len(title)))
                category_token.append([self.vocab[vert]])
                subcategory_token.append([self.vocab[subvert]])


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
        """ 
            init behavior logs given behaviors file.
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

    def parser_one_line(self, line):
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
            impr_label = self.labels[line]
            impr = self.imprs[line]

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

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_category_index,
                    candidate_subcategory_index,
                    click_title_index,
                    click_category_index,
                    click_subcategory_index
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

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
                click_title_index = self.news_title_array[self.histories[line]]
                click_category_index = self.news_category_array[self.histories[line]]
                click_subcategory_index = self.news_subcategory_array[self.histories[line]]

                impr_index = self.impr_indexes[line]
                user_index.append(self.uindexes[line])
                
                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_category_index,
                    candidate_subcategory_index,
                    click_title_index,
                    click_category_index,
                    click_subcategory_index
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
        impr_indexes = []
        user_indexes = []

        candidate_title_indexes = []
        candidate_category_indexes = []
        candidate_subcategory_indexes = []

        click_title_indexes = []
        click_category_indexes = []
        click_subcategory_indexes = []

        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                label,
                impr_index,
                user_index,
                candidate_title_index,
                candidate_category_index,
                candidate_subcategory_index,
                click_title_index,
                click_category_index,
                click_subcategory_index
            ) in self.parser_one_line(index):

                # append one log in the batch
                candidate_title_indexes.append(candidate_title_index)
                candidate_category_indexes.append(candidate_category_index)
                candidate_subcategory_indexes.append(candidate_subcategory_index)

                click_title_indexes.append(click_title_index)
                click_category_indexes.append(click_category_index)
                click_subcategory_indexes.append(click_subcategory_index)

                impr_indexes.append(impr_index)
                user_indexes.append(user_index)
                label_list.append(label)

                cnt += 1
                if cnt >= self.batch_size:
                    
                    yield self._convert_data(
                        label_list,
                        impr_indexes,
                        user_indexes,
                        candidate_title_indexes,
                        candidate_category_indexes,
                        candidate_subcategory_indexes,
                        click_title_indexes,
                        click_category_indexes,
                        click_subcategory_indexes
                    )
                    label_list = []
                    impr_indexes = []
                    user_indexes = []

                    candidate_title_indexes = []
                    candidate_category_indexes = []
                    candidate_subcategory_indexes = []

                    click_title_indexes = []
                    click_category_indexes = []
                    click_subcategory_indexes = []
                    cnt = 0

        "in case the last few examples can't fill a batch, abandon them"

        # if cnt > 0:
        #     yield self._convert_data(
        #         label_list,
        #         impr_indexes,
        #         user_indexes,
        #         candidate_title_indexes,
        #         click_title_indexes,
        #     )

    def _convert_data(
        self,
        label_list,
        impr_indexes,
        user_indexes,
        candidate_title_indexes,
        candidate_category_indexes,
        candidate_subcategory_indexes,
        click_title_indexes,
        click_category_indexes,
        click_subcategory_indexes
        ):
        """Convert data of one candidate into torch.tensor that are good for further model operation, 
        
        Args:
            label_list (list): a list of ground-truth labels.
            impr_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.
            
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        labels = torch.tensor(label_list, dtype=torch.float32,device=self.device)
        # impr_indexes = torch.tensor(impr_indexes, dtype=torch.int64,device=self.device)
        user_indexes = torch.tensor(user_indexes, dtype=torch.int64,device=self.device)

        # candidate_title_index_batch = torch.cat(candidate_title_indexes,dim=0).unsqueeze(dim=0)
        # click_title_index_batch = torch.cat(click_title_indexes,dim=0).unsqueeze(dim=0)
        candidate_title_index_batch = torch.tensor(
            candidate_title_indexes, dtype=torch.int64,device=self.device
        )
        candidate_category_index_batch = torch.tensor(
            candidate_category_indexes, dtype=torch.int64,device=self.device
        )
        candidate_subcategory_index_batch = torch.tensor(
            candidate_subcategory_indexes, dtype=torch.int64,device=self.device
        )

        click_title_index_batch = torch.tensor(click_title_indexes, dtype=torch.int64,device=self.device)
        click_category_index_batch = torch.tensor(click_category_indexes, dtype=torch.int64,device=self.device)
        click_subcategory_index_batch = torch.tensor(click_subcategory_indexes, dtype=torch.int64,device=self.device)

        return {
            "impression_index_batch": impr_indexes,
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_category_batch":click_category_index_batch,
            "clicked_subcategory_batch":click_subcategory_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_category_batch": candidate_category_index_batch,
            "candidate_subcategory_batch": candidate_subcategory_index_batch,
            "labels": labels,
        }

    # def load_user_from_file(self, news_file, behavior_file):
    #     """Read and parse user data from news file and behavior file, generate batch_size of user examples, 
    #     each of which contains a user id, an impression id, a union tensor of history clicked news' title tensor
        
    #     Args:
    #         news_file (str): A file contains several informations of news.
    #         beahaviros_file (str): A file contains information of user impressions.

    #     Returns:
    #         obj: An iterator that will yields parsed user feature, in the format of dict.
    #     """

    #     if not hasattr(self, "news_title_array"):
    #         self.init_news(news_file)

    #     if not hasattr(self, "impr_indexes"):
    #         self.init_behaviors(behavior_file)

    #     user_indexes = []
    #     impr_indexes = []
    #     click_title_indexes = []
    #     cnt = 0

    #     for index in range(len(self.impr_indexes)):
    #         click_title_indexes.append(self.news_title_array[self.histories[index]])
    #         user_indexes.append(self.uindexes[index])
    #         impr_indexes.append(self.impr_indexes[index])

    #         cnt += 1
    #         if cnt >= self.batch_size:
    #             yield self._convert_user_data(
    #                 user_indexes, impr_indexes, click_title_indexes,
    #             )
    #             user_indexes = []
    #             impr_indexes = []
    #             click_title_indexes = []
    #             cnt = 0

    #     if cnt > 0:
    #         yield self._convert_user_data(
    #             user_indexes, impr_indexes, click_title_indexes,
    #         )

    # def _convert_user_data(self, user_indexes, impr_indexes, click_title_indexes):
    #     """Convert data into numpy arrays that are good for further model operation.
        
    #     Args:
    #         user_indexes (list): a list of user indexes.
    #         click_title_indexes (list): words indices for user's clicked news titles.
            
    #     Returns:
    #         dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
    #     """

    #     user_indexes = torch.tensor(user_indexes, dtype=torch.int32)
    #     impr_indexes = torch.tensor(impr_indexes, dtype=torch.int32)
    #     click_title_index_batch = torch.tensor(click_title_indexes, dtype=torch.int64)

    #     return {
    #         "user_index_batch": user_indexes,
    #         "impr_index_batch": impr_indexes,
    #         "clicked_title_batch": click_title_index_batch,
    #     }

    # def load_news_from_file(self, news_file):
    #     """Read and parse user data from news file.
        
    #     Args:
    #         news_file (str): A file contains several informations of news.
            
    #     Returns:
    #         obj: An iterator that will yields parsed news feature, in the format of dict.
    #     """
    #     if not hasattr(self, "news_title_array"):
    #         self.init_news(news_file)

    #     news_indexes = []
    #     candidate_title_indexes = []
    #     cnt = 0

    #     for index in range(len(self.news_title_array)):
    #         news_indexes.append(index)
    #         candidate_title_indexes.append(self.news_title_array[index])

    #         cnt += 1
    #         if cnt >= self.batch_size:
    #             yield self._convert_news_data(
    #                 news_indexes, candidate_title_indexes,
    #             )
    #             news_indexes = []
    #             candidate_title_indexes = []
    #             cnt = 0

    #     if cnt > 0:
    #         yield self._convert_news_data(
    #             news_indexes, candidate_title_indexes,
    #         )
    
    # def _convert_news_data(self, news_indexes, candidate_title_indexes):
    #     """Convert data into numpy arrays that are good for further model operation.
        
    #     Args:
    #         news_indexes (list): a list of news indexes.
    #         candidate_title_indexes (list): the candidate news titles' words indices.
            
    #     Returns:
    #         dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
    #     """
        
    #     news_indexes_batch = torch.tensor(news_indexes, dtype=torch.int32)
    #     candidate_title_index_batch = torch.tensor(
    #         candidate_title_indexes, dtype=torch.int32
    #     )

    #     return {
    #         "news_index_batch": news_indexes_batch,
    #         "candidate_title_batch": candidate_title_index_batch,
    #     }

    # def load_impression_from_file(self, behaivors_file):
    #     """Read and parse impression data from behaivors file.
        
    #     Args:
    #         behaivors_file (str): A file contains several informations of behaviros.
            
    #     Returns:
    #         obj: An iterator that will yields parsed impression data, in the format of dict.
    #     """
    #     if not hasattr(self, "histories"):
    #         self.init_behaviors(behaivors_file)
    #     indexes = np.arange(len(self.labels))

    #     for index in indexes:
    #         impr_label = torch.tensor(self.labels[index], dtype=torch.int32)
    #         impr_news = torch.tensor(self.imprs[index], dtype=torch.int32)

    #         yield (
    #             self.impr_indexes[index],
    #             impr_news,
    #             self.uindexes[index],
    #             impr_label,
    #         )