import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
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
        attn_weights = torch.bmm(query,key)
        attn_output = torch.bmm(attn_weights,value)/math.sqrt(self.embedding_dim)
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
    
    def _multi_head_self_attention(self,input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
        
        Returns:
            multi_head_self_attn: tensor of [batch_size, *, signal_length, embedding_dim]
        """
        self_attn_outputs = [self._self_attention(input,i) for i in range(self.head_num)]
        multi_head_self_attns = torch.cat(self_attn_outputs,dim=2)
        multi_head_self_attn = self.feedForward(multi_head_self_attns).view(self.batch_size,-1,self.signal_length,self.embedding_dim)
        assert multi_head_self_attn.shape[1] == input.shape[1]
        return multi_head_self_attn
    
    def forward(self,x):
        return self._multi_head_self_attention(x)