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
        self.transformer_length = hparams['title_size'] * 2 + 1
        self.his_size = hparams['his_size']

        self.head_num = hparams['head_num']
        self.query_dim = hparams['query_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.repr_dim = self.head_num * self.value_dim
        
        self.query_words = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))
        self.query_itrs = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))

        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])
        self.CosSim = nn.CosineSimilarity(dim=-1)

        self.queryProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.keyProject_words = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)

        self.queryProject_itrs = nn.ModuleList([]).extend([nn.Linear(self.repr_dim,self.repr_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_itrs = nn.ModuleList([]).extend([nn.Linear(self.repr_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.keyProject_itrs = nn.Linear(self.repr_dim, self.query_dim, bias=True)

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

    def _self_attention(self,input,head_idx,mode):
        """ apply self attention of head#idx over input tensor
        
        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        if mode==1:
            query = self.queryProject_words[head_idx](input)
            attn_output = self._scaled_dp_attention(query,input,input)
            self_attn_output = self.valueProject_words[head_idx](attn_output)
        elif mode==2:
            query = self.queryProject_itrs[head_idx](input)
            attn_output = self._scaled_dp_attention(query,input,input)
            self_attn_output = self.valueProject_itrs[head_idx](attn_output)

        return self_attn_output

    def _multi_head_self_attention(self,input,mode):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length/transformer_length, repr_dim]
        
        Returns:
            additive_attn_repr: tensor of [batch_size, *, repr_dim]
            multi_head_self_attn_value: tensor of [batch_size, *, signal_length, repr_dim]

        """
        if mode == 1:
            self_attn_outputs = [self._self_attention(input,i,1) for i in range(self.head_num)]
            multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
            # project the embedding of each words to query subspace
            # keep the original embedding of each words as values
            multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))
            return multi_head_self_attn_value

        elif mode == 2:
            self_attn_outputs = [self._self_attention(input,i,2) for i in range(self.head_num)]
            multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
            multi_head_self_attn_key = torch.tanh(self.keyProject_itrs(multi_head_self_attn_value))
            additive_attn_repr = self._scaled_dp_attention(self.query_itrs,multi_head_self_attn_key,multi_head_self_attn_value).squeeze(dim=-2)
            return additive_attn_repr

    def _fusion(self, cdd_news_embedding, his_news_embedding):
        """ concatenate candidate news title and history news title
        
        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, repr_dim] 
            his_news_embedding: tensor of [batch_size, his_size, signal_length, repr_dim] 

        Returns:
            fusion_news_embedding: tensor of [batch_size, cdd_size, his_size, transformer_length, repr_dim]
        """
        fusion_news_embedding = torch.zeros((self.batch_size, self.cdd_size, self.his_size, self.transformer_length, self.repr_dim) ,device=self.device)
        fusion_news_embedding[:,:,:,:self.signal_length,:] = cdd_news_embedding.unsqueeze(dim=2)
        fusion_news_embedding[:,:,:,(self.signal_length + 1):] = his_news_embedding.unsqueeze(dim=1)
        # split two news with <PAD>
        fusion_news_embedding[:,:,:,self.signal_length] = 1.
        return fusion_news_embedding

    def _news_encoder(self,news_batch):
        """ encode batch of news with Multi-Head Self-Attention
        
        Args:
            news_batch: tensor of [batch_size, *, signal_length]
            word_query: tensor of [set_size, preference_dim]
        
        Returns:
            news_embedding_attn: tensor of [batch_size, *, signal_length, repr_dim] 
        """
        news_embedding = self.DropOut(self.embedding[news_batch])
        news_embedding_attn = self._multi_head_self_attention(news_embedding,1)
        return news_embedding_attn

    def _fusion_transform(self,fusion_news_embedding):
        """ encode fused news into embeddings
        
        Args:
            fusion_news_embedding: tensor of [batch_size, cdd_size, his_size, transformer_length, repr_dim]
        
        Returns:
            fusion_vectors: tensor of [batch_size, cdd_size, repr_dim]
        """
        fusion_vectors = self._multi_head_self_attention(fusion_news_embedding, mode=2)#.view(self.batch_size, self.cdd_size, -1)
        fusion_vectors = torch.mean(fusion_vectors, dim=2)
        return fusion_vectors
    
    def _click_predictor(self,fusion_vectors):
        """ calculate batch of click probability              
        Args:
            pooling_vectors: tensor of [batch_size, cdd_size, kernel_num]
        
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
        cdd_news_embedding = self._news_encoder(x['candidate_title'].long().to(self.device))
        his_news_embedding = self._news_encoder(x['clicked_title'].long().to(self.device))
        fusion_news_embedding = self._fusion(cdd_news_embedding, his_news_embedding)
        fusion_vectors = self._fusion_transform(fusion_news_embedding)
        score_batch = self._click_predictor(fusion_vectors)
        return score_batch