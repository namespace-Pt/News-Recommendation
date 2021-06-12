import torch
import math
import torch.nn as nn
from models.base_model import BaseModel

class NPAModel(BaseModel):
    def __init__(self,hparams,encoder):
        super().__init__(hparams)
        self.name = 'npa'

        self.signal_length = hparams['title_size']

        self.encoder = encoder

        self.hidden_dim = self.encoder.hidden_dim
        self.preference_dim =self.encoder.query_dim
        self.user_dim = self.encoder.user_dim

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = self.encoder.user_embedding
        # project e_u to word news preference vector of preference_dim
        self.newsPrefProject = nn.Linear(self.user_dim,self.preference_dim)
        # project preference query to vector of filter_num
        self.newsQueryProject = nn.Linear(self.preference_dim,self.hidden_dim)

        self.RELU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.Tanh = nn.Tanh()
        self.DropOut = self.encoder.DropOut

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _user_encoder(self,his_news_batch,news_query,word_query):
        """ encode batch of user history clicked news to user representations of [batch_size,filter_num]

        Args:
            his_news_batch: tensor of [batch_size, his_size, title_size]
            news_query: tensor of [batch_size, preference_dim]
            word_query: tensor of [batch_size, preference_dim]

        Returns:
            user_repr: tensor of [batch_size, filter_num]
        """
        his_news_reprs = self._news_encoder(his_news_batch,word_query).view(self.batch_size,self.his_size,self.filter_num).permute(0,2,1)
        user_reprs = self._attention_news(news_query,his_news_reprs)

        return user_reprs

    def _click_predictor(self,cdd_news_repr,user_repr):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim]
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = torch.bmm(cdd_news_repr,user_repr.transpose(-2,-1)).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        user_index = x['user_index'].long().to(self.device)

        cdd_news = x['candidate_title'].long().to(self.device)
        _, cdd_news_repr = self.encoder(
            cdd_news,
            user_index=user_index)

        his_news = x['clicked_title'].long().to(self.device)
        _, his_news_repr = self.encoder(
            his_news,
            user_index=user_index)

        e_u = self.DropOut(self.user_embedding(user_index))
        news_query = self.Tanh(self.newsQueryProject(
            self.RELU(self.newsPrefProject(e_u))))

        user_repr = self._scaled_dp_attention(news_query, his_news_repr, his_news_repr)
        score = self._click_predictor(cdd_news_repr, user_repr)
        return score