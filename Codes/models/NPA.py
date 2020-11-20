'''
Author: Pt
Date: 2020-11-05 18:05:03
LastEditTime: 2020-11-20 10:24:33
'''

import torch
import torch.nn as nn

class NPAModel(nn.Module):
    def __init__(self,hparams,vocab,npratio):
        super().__init__()
        
        self.npratio = npratio
        self.dropout_p = hparams['dropout_p']
        self.metrics = hparams['metrics']

        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size =hparams['his_size']

        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.preference_dim =hparams['preference_dim']

        self.device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device('cpu')

        # pretrained embedding
        self.embedding = vocab.vectors
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.functional.softmax

        # project userID to dense vector e_u of user_dim
        self.userProject = nn.Linear(1,self.user_dim)
        # project e_u to word query preference vector of preference_dim
        self.wordQueryProject = nn.Linear(self.user_dim,self.preference_dim)
        # project e_u to word news preference vector of preference_dim
        self.newsQueryProject = nn.Linear(self.user_dim,self.preference_dim)
        # project preference query to vector of filter_num
        self.wordPrefProject = nn.Linear(self.preference_dim,self.filter_num)
        self.newsPrefProject = nn.Linear(self.preference_dim,self.filter_num)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size=3,padding=1)
        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=self.dropout_p)

    def _user_projection(self,user_index_batch):
        """ project user ID to dense vector e_u of [batch_size,user_dim] and store it for further use
        
        Args:
            x: one training batch example           
        """
        e_u = self.userProject(user_index_batch)
        if self.dropout_p > 0:
            e_u = self.DropOut(e_u)
        self.e_u = e_u
    
    def _word_query_projection(self):
        """ project e_u to word preference query vector of [batch_size,preference_dim]
        
        Returns:
            word_query: tensor of batch_size * preference_dim       
        """
        word_query = self.wordQueryProject(self.e_u)
        word_query = self.RELU(word_query)
        if self.dropout_p > 0:
            word_query = self.DropOut(word_query)

        return word_query
    
    def _news_query_projection(self):
        """ project e_u to news preference query vector of [batch_size,preference_dim]
        
        Returns:
            news_query: tensor of batch_size * preference_dim       
        """
        news_query = self.newsQueryProject(self.e_u)
        news_query = self.RELU(news_query)
        if self.dropout_p > 0:
            news_query = self.DropOut(news_query)
        
        return news_query

    def _attention_word(self,query,keys):
        """ apply original attention mechanism
            first unsqueeze the query for computation,
            then use inner-product to get batch of attn_results,
            then use softmax to get normalized weights of each title of each example in the batch,
            finally multiply batch of keys(title emebdding) by attn_weights
        
        Args:
            query: tensor of batch_size * preference_dim
            keys: tensor of batch_size * filter_num * title_size
        
        Returns:
            attn_aggr: tensor of batch_size * filter_num, (batch of news embedding)
        """

        # return tensor of batch_size * 1 * filter_num
        query = self.wordPrefProject(query).unsqueeze(dim=1)
    
        # return tensor of batch_size * 1 * title_size
        attn_results = torch.bmm(query,keys)
        # return tensor of batch_size * title_size * 1
        attn_weights = self.softmax(attn_results,dim=2).permute(0,2,1)
        
        # return tensor of batch_size * filter_num
        attn_aggr = torch.bmm(keys,attn_weights).squeeze()
        return attn_aggr

    def _attention_news(self,query,keys):
        """ apply original attention mechanism
            first unsqueeze the query for computation,
            then use inner-product to get batch of attn_results,
            then use softmax to get normalized weights of each news in the batch,
            finally multiply batch of keys(news emebdding) by attn_weights
        
        Args:
            query: tensor of batch_size * preference_dim
            keys: tensor of batch_size * filter_num * his_size
        
        Returns:
            attn_aggr: tensor of batch_size * filter_num, (batch of user embedding)
        """

        # return tensor of batch_size * 1 * filter_num
        query = self.newsPrefProject(query).unsqueeze(dim=1)
    
        # return tensor of batch_size * 1 * his_size
        attn_results = torch.bmm(query,keys)
        # return tensor of batch_size * title_size * 1
        attn_weights = self.softmax(attn_results,dim=2).permute(0,2,1)
        
        # return tensor of batch_size * filter_num
        attn_aggr = torch.bmm(keys,attn_weights).squeeze()
        return attn_aggr

    def _news_encoder(self,news_set,word_query):
        """ encode set of news to news representations of [set_size,filter_num]
        
        Args:
            news_set: tensor of set_size * title_size
            word_query: tensor of set_size * preference_dim      
        
        Returns:
            news_repr: tensor of set_size * filter_num 
        """

        # important not to directly apply view function
        # return tensor of set_size * embedding_dim * title_size
        cdd_title_embedding = self.embedding[news_set].permute(0,2,1).to(self.device)
        
        # return tensor of set_size * filter_num * title_size
        cdd_title_embedding = self.CNN(cdd_title_embedding)
        cdd_title_embedding = self.RELU(cdd_title_embedding)
        if self.dropout_p > 0:
            cdd_title_embedding = self.DropOut(cdd_title_embedding)
        
        news_repr = self._attention_word(word_query,cdd_title_embedding)

        return news_repr

    def _user_encoder(self,his_news_batch,news_query,word_query):
        """ encode batch of user history clicked news to user representations of [batch_size,filter_num]
        
        Args:
            his_news_batch: tensor of batch_size * his_size * title_size
            news_query: tensor of batch_size * preference_dim
            word_query: tensor of batch_size * preference_dim     
        
        Returns:
            user_repr: tensor of batch_size * filter_num 
        """
        user_reprs = torch.zeros((self.batch_size,self.filter_num),dtype=torch.float32,device=self.device)
        for i in range(self.batch_size):
            his_news_repr = self._news_encoder(his_news_batch[i],word_query[i].expand(self.his_size,-1))
            user_repr = self._attention_news(news_query[i].unsqueeze(dim=0),his_news_repr.unsqueeze(dim=0).permute(0,2,1))
            user_reprs[i] = user_repr
        
        return user_reprs

    def _click_predictor(self,cdd_news_repr,user_repr):
        """ calculate batch of click probability
        
        Args:
            cdd_news_repr: tensor of batch_size * npratio+1/1 * filter_num
            user_repr: tensor of batch_size * 1 * filter_num
        
        Returns:
            score: tensor of batch_size * npratio+1/1
        """
        score = torch.bmm(cdd_news_repr,user_repr.permute(0,2,1))
        if self.npratio > 0:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score.squeeze()

    def forward(self,x):
        self._user_projection(x['user_index'].float().to(self.device))
        word_query = self._word_query_projection()
        news_query = self._news_query_projection()
        cdd_news_batch = x['candidate_title'].long().to(self.device)

        if self.npratio > 0:
            cdd_news_reprs = torch.zeros((self.batch_size,self.npratio+1,self.filter_num),dtype=torch.float32,device=self.device)

            for i in range(self.batch_size):
                cdd_news_repr = self._news_encoder(cdd_news_batch[i],word_query[i].expand(self.npratio+1,-1))
                cdd_news_reprs[i] = cdd_news_repr
        else:
            cdd_news_reprs = self._news_encoder(cdd_news_batch.squeeze(),word_query).unsqueeze(dim=1)
        
        user_repr = self._user_encoder(x['clicked_title'].long().to(self.device),news_query,word_query)
        
        score = self._click_predictor(cdd_news_reprs.view(self.batch_size,-1,self.filter_num),user_repr.unsqueeze(dim=1))
        return score