import math
import torch
import torch.nn as nn

class NRMSModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']
        
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.metrics = hparams['metrics']
        self.device = torch.device(hparams['device'])
        self.embedding = vocab.vectors.to(self.device)

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']

        self.head_num = hparams['head_num']
        self.query_dim = hparams['query_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.repr_dim = self.head_num * self.value_dim
        
        self.query_words = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))
        self.query_news = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))

        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.queryProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim, bias=False) for _ in range(self.head_num)])
        self.queryProject_news = nn.ModuleList([]).extend([nn.Linear(self.repr_dim,self.repr_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_news = nn.ModuleList([]).extend([nn.Linear(self.repr_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.keyProject_words = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.keyProject_news = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.learningToRank = nn.Linear(self.repr_dim, 1)

    def _scaled_dp_attention(self,query,key,value):
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
        key = key.transpose(-2,-1)

        attn_weights = torch.matmul(query,key)/math.sqrt(self.embedding_dim)
        attn_weights = self.softmax(attn_weights)
        
        attn_output = torch.matmul(attn_weights,value)

        return attn_output

    def _self_attention(self,input,head_idx,mode):
        """ apply self attention of head#idx over input tensor
        
        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index
            mode: 0/1, 1 for words self attention, 0 for news self attention

        Returns:
            self_attn_output: tensor of [batch_size, *, embedding_dim]
        """
        if mode:
            query = self.queryProject_words[head_idx](input)

            attn_output = self._scaled_dp_attention(query,input,input)
            self_attn_output = self.valueProject_words[head_idx](attn_output)

            return self_attn_output

        else:
            query = self.queryProject_news[head_idx](input)

            attn_output = self._scaled_dp_attention(query,input,input)
            self_attn_output = self.valueProject_news[head_idx](attn_output)

            return self_attn_output
    
    def _word_attention(self, query, key, value):
        """ apply word-level attention

        Args:
            attn_word_embedding_key: tensor of [batch_size, *, signal_length, query_dim]
            attn_word_embedding_value: tensor of [batch_size, *, signal_length, repr_dim]
        
        Returns:
            attn_output: tensor of [batch_size, *, repr_dim]
        """
        query = query.expand(key.shape[0], key.shape[1], 1, self.query_dim)

        attn_output = self._scaled_dp_attention(query,key,value).squeeze(dim=2)

        return attn_output

    def _news_attention(self, query, key, value):
        """ apply news-level attention

        Args:
            attn_word_embedding_key: tensor of [batch_size, his_size, query_dim]
            attn_word_embedding_value: tensor of [batch_size, his_size, repr_dim]
        
        Returns:
            attn_output: tensor of [batch_size, 1, repr_dim]
        """
        query = query.expand(key.shape[0], 1, self.query_dim)

        attn_output = self._scaled_dp_attention(query,key,value)

        return attn_output


    def _multi_head_self_attention(self,input,mode):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
            mode: 0/1, 1 for words self attention, 0 for news self attention
        
        Returns:
            multi_head_self_attn: tensor of [batch_size, *, 1, repr_dim]
        """
        if mode:
            self_attn_outputs = [self._self_attention(input,i,1) for i in range(self.head_num)]

            # project the embedding of each words to query subspace
            # keep the original embedding of each words as values
            multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
            multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))

            additive_attn_embedding = self._word_attention(self.query_words, multi_head_self_attn_key,multi_head_self_attn_value)

            return additive_attn_embedding

        else:
            self_attn_outputs = [self._self_attention(input,i,0) for i in range(self.head_num)]

            # project the embedding of each words to query subspace
            # keep the original embedding of each words as values
            multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
            multi_head_self_attn_key = torch.tanh(self.keyProject_news(multi_head_self_attn_value))

            additive_attn_embedding = self._news_attention(self.query_news, multi_head_self_attn_key,multi_head_self_attn_value)

            return additive_attn_embedding

    def _news_encoder(self,news_batch):
        """ encode set of news to news representations of [batch_size, cdd_size, tranformer_dim]
        
        Args:
            news_batch: tensor of [batch_size, cdd_size, title_size]
        
        Returns:
            news_reprs: tensor of [batch_size, cdd_size, repr_dim]
        """
        news_embedding = self.DropOut(self.embedding[news_batch])
        news_reprs = self._multi_head_self_attention(news_embedding,1)
        return news_reprs
    
    def _user_encoder(self,his_news_batch):
        """ encode set of news to news representations of [batch_size, cdd_size, tranformer_dim]
        
        Args:
            his_news_batch: tensor of [batch_size, his_size, title_size]
        
        Returns:
            user_reprs: tensor of [batch_size, 1, repr_dim]
        """
        his_news_reprs = self._news_encoder(his_news_batch)
        user_reprs = self._multi_head_self_attention(his_news_reprs,0)
        return user_reprs

    def _click_predictor(self,cdd_news_repr,user_repr):
        """ calculate batch of click probability
        
        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, repr_dim]
            user_repr: tensor of [batch_size, 1, repr_dim]
        
        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = torch.bmm(cdd_news_repr,user_repr.permute(0,2,1))
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score.squeeze()

    def forward(self,x):
        cdd_news_reprs = self._news_encoder(x['candidate_title'].long().to(self.device))
        user_reprs = self._user_encoder(x['clicked_title'].long().to(self.device))

        score = self._click_predictor(cdd_news_reprs,user_reprs)

        return score