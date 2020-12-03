import math
import torch
import torch.nn as nn

class NRMSEncoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.device = torch.device(hparams['gpu']) if torch.cuda.is_available() else torch.device('cpu')

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.head_num = hparams['head_num']
        self.embedding_dim = hparams['embedding_dim']
        self.transformer_dim = hparams['transformer_dim']

        self.queryProject = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim) for _ in range(self.head_num)])
        self.valueProject = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.transformer_dim) for _ in range(self.head_num)])

        self.feedForward = nn.Linear(self.transformer_dim * self.head_num, self.embedding_dim)
    
    def _scaled_dp_attention(self,query,key,value):
        """ calculate scaled attended output of values
        
        Args:
            query: tensor of [set_size, signal_length, key_dim(embedding_dim)]
            key: tensor of [set_size, signal_length, key_dim(embedding_dim)]
            value: tensor of [set_size, signal_length, value_dim(embedding_dim)]
        
        Returns:
            attn_output: tensor of [set_size, signal_length, value_dim(embedding_dim)]
        """
        attn_weights = torch.bmm(query,key)/math.sqrt(self.embedding_dim)
        attn_weights = self.softmax(attn_weights,dim=2)
        
        attn_output = torch.bmm(attn_weights,value)
        return attn_output

    def _self_attention(self,input,head_idx):
        """ apply self attention of head#idx over input tensor
        
        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
        
        Returns:
            self_attn_output: tensor of [batch_size, *, signal_length, embedding_dim]
        """
        query = self.queryProject[head_idx](input).view(-1,self.signal_length,self.embedding_dim)
        key = input.view(-1,self.signal_length,self.embedding_dim).permute(0,2,1)
        value = input.view(-1,self.signal_length,self.embedding_dim)

        attn_output = self._scaled_dp_attention(query,key,value)
        self_attn_output = self.valueProject[head_idx](attn_output)

        return self_attn_output
    
    def _word_attention(self, attn_word_embedding_key, attn_word_embedding_value):
        """ apply word-level attention

        Args:
            query: tensor of [1, query_dim]
            attn_word_embedding_key: tensor of [set_size, signal_length, query_dim]
            attn_word_embedding_value: tensor of [set_size, signal_length, head_num * transformer_dim]
        
        Returns:
            attn_output: tensor of [set_size, query_dim]
        """
        query = self.query.expand(attn_word_embedding_key.shape[0],self.query_dim).unsqueeze(dim=1)
        key = attn_word_embedding_key.permute(0,2,1)

        attn_weights = torch.bmm(query,key)/math.sqrt(self.query_dim)
        attn_weights = self.softmax(attn_weights,dim=2)
        attn_output = torch.bmm(attn_weights,attn_word_embedding_value).squeeze()
        
        return attn_output


    def _multi_head_self_attention(self,input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
        
        Returns:
            multi_head_self_attn: tensor of [batch_size, *, signal_length, embedding_dim]
        """
        self_attn_outputs = [self._self_attention(input,i) for i in range(self.head_num)]

        # here I'm strictly following the paper, however I argue it is neither necessary nor plausible to 
        # project concatenation of self-attended vector to a new subspace for further attending while keep 
        # the origin as attention value
        multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=2)
        multi_head_self_attn_key = torch.nn.functional.tanh(self.feedForward(multi_head_self_attn_value))

        assert multi_head_self_attn_key.shape[0] == input.shape[0] * input.shape[1]

        additive_attn_embedding = self._word_attention(multi_head_self_attn_key,multi_head_self_attn_value)

        return additive_attn_embedding

    def _news_encoder(self,news_batch):
        """ encode set of news to news representations of [batch_size, his_size/npratio+1/1, tranformer_dim]
        
        Args:
            news_batch: tensor of [batch_size, his_size/npratio+1/1, title_size]
            word_query: tensor of [set_size, preference_dim]
        
        Returns:
            news_repr_attn: tensor of [batch_size, 1/npratio+1/his_size, transformer_dim]
            news_repr_origin: tensor of [batch_size, 1/npratio+1/his_size, signal_length, embedding_dim] 
        """
        news_embedding_origin = self.embedding[news_batch].to(self.device)
        news_embedding_attn = self._multi_head_self_attention(news_embedding_origin).view(self.batch_size,-1,self.repr_dim)
        return news_embedding_attn, news_embedding_origin
    
    def forward(self,x):
        return self._news_encoder(x)