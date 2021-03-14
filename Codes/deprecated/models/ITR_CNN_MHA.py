import torch
import torch.nn as nn

class GCAModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1

        self.device = torch.device(hparams['device'])
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']

        self.dropout_p = hparams['dropout_p']

        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
       
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])
        
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size=3,padding=1)
        self.learningToRank = nn.Linear(self.repr_dim, 1)
        # self.learningToRank = nn.Linear(self.repr_dim * self.his_size, 1)

    def _scaled_dp_attention(self,query,key,value):
        """ calculate scaled attended output of values
        
        Args:
            query: tensor of [*, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]
        
        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2,-1)

        attn_weights = torch.matmul(query,key)/torch.sqrt(torch.tensor([self.embedding_dim],dtype=torch.float,device=self.device))
        attn_weights = self.softmax(attn_weights)
        
        attn_output = torch.matmul(attn_weights,value)

        return attn_output.squeeze(dim=-2)

    def _self_attention(self,input,head_idx):
        """ apply self attention of head#idx over input tensor
        
        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        query = self.queryProject_words[head_idx](input)

        attn_output = self._scaled_dp_attention(query,input,input)
        self_attn_output = self.valueProject_words[head_idx](attn_output)

        return self_attn_output
    
    def _word_attention(self, query, key, value):
        """ apply word-level attention

        Args:
            query: tensor of [1, query_dim]
            key: tensor of [batch_size, *, transformer_length, query_dim]
            value: tensor of [batch_size, *, transformer_length, repr_dim]

        Returns:
            attn_output: tensor of [batch_size, *, repr_dim]
        """
        # query = query.expand(key.shape[0], key.shape[1], key.shape[2], 1, self.query_dim)

        attn_output = self._scaled_dp_attention(query,key,value).squeeze(dim=-2)

        return attn_output.squeeze(dim=-2)


    def _multi_head_self_attention(self,input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, transformer_length, repr_dim]
        
        Returns:
            multi_head_self_attn: tensor of [batch_size, *, repr_dim]
        """
        self_attn_outputs = [self._self_attention(input,i) for i in range(self.head_num)]

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
        multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))

        additive_attn_repr = self._word_attention(self.query_words, multi_head_self_attn_key, multi_head_self_attn_value)
        return additive_attn_repr

    def _fusion(self, cdd_news, his_news):
        """ concatenate candidate news title and history news title
        
        Args:
            cdd_news: tensor of [batch_size, cdd_size, signal_length] 
            his_news: tensor of [batch_size, his_size, signal_length] 

        Returns:
            fusion_news: tensor of [batch_size, cdd_size, his_size, transformer_length]
        """
        fusion_news = torch.zeros((self.batch_size, self.cdd_size, self.his_size, self.transformer_length) ,device=self.device).long()
        fusion_news[:,:,:,:self.signal_length] = cdd_news.unsqueeze(dim=2)
        fusion_news[:,:,:,(self.signal_length + 1):] = his_news.unsqueeze(dim=1)
        # split two news with <PAD>
        fusion_news[:,:,:,self.signal_length] = 1
        return fusion_news
    
    def _news_encoder(self,news_batch):
        """ capture local text
        
        Args:
            news_batch: tensor of [batch_size, cdd_size, his_size, transformer_length]
        
        Returns:
            news_emebdding: tensor of [batch_size, cdd_size, his_size, transformer_length, filter_num] 
        """

        news_embedding = self.embedding[news_batch].transpose(-2,-1).view(-1,self.embedding_dim,self.transformer_length)
        
        news_embedding = self.CNN(news_embedding).transpose(-2,-1).view(self.batch_size, self.cdd_size, self.his_size, self.transformer_length, self.filter_num)
        news_embedding = self.ReLU(news_embedding)

        if self.dropout_p > 0:
            news_embedding = self.DropOut(news_embedding)

        return news_embedding

    def _fusion_transform(self,fusion_news_embedding):
        """ encode fused news into embeddings
        
        Args:
            fusion_news_embedding: tensor of [batch_size, cdd_size, his_size, transformer_length, filter_num]
        
        Returns:
            fusion_vectors: tensor of [batch_size, cdd_size, repr_dim]
        """
        fusion_vectors = self._multi_head_self_attention(fusion_news_embedding)#.view(self.batch_size, self.cdd_size, -1)
        fusion_vectors = torch.mean(fusion_vectors, dim=-2)

        return fusion_vectors
    
    def _click_predictor(self,fusion_vectors):
        """ calculate batch of click probability              
        Args:
            fusion_vectors: tensor of [batch_size, cdd_size, repr_dim]
        
        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = self.learningToRank(fusion_vectors).squeeze(dim=-1)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        fusion_news = self._fusion(x['candidate_title'].long().to(self.device), x['clicked_title'].long().to(self.device))
        fusion_news_embedding = self._news_encoder(fusion_news)
        fusion_vectors = self._fusion_transform(fusion_news_embedding)
        score_batch = self._click_predictor(fusion_vectors)
        return score_batch