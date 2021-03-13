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

        self.embedding_dim = hparams['embedding_dim']
        self.head_num = hparams['head_num']
        self.repr_dim = self.head_num * hparams['value_dim']

        self.tfmEncoder = nn.TransformerEncoderLayer(d_model=self.repr_dim, nhead=self.head_num)
        # self.tfmInteractor = nn.TransformerEncoderLayer(d_model=self.repr_dim, nhead=self.head_num)

        self.query_itr = nn.Parameter(torch.rand(1,self.repr_dim), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        # self.learningToRank = nn.Linear(self.repr_dim, 1)
        self.learningToRank = nn.Linear(self.his_size, 1) 
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

    def _news_encoder(self,news_batch):
        """ encode batch of news with Multi-Head Self-Attention
        
        Args:
            news_batch: tensor of [batch_size, *, signal_length]
            word_query: tensor of [set_size, preference_dim]
        
        Returns:
            news_embedding_attn: tensor of [batch_size, *, signal_length, repr_dim] 
        """
        news_embedding = self.embedding[news_batch].view(-1, self.signal_length, self.embedding_dim)
        news_embedding_attn = self.tfmEncoder(news_embedding)
        return news_embedding_attn.view(self.batch_size, news_batch.shape[1], self.signal_length, self.repr_dim)

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
    
    def _fusion_transform(self,fusion_news_embedding):
        """ encode fused news into embeddings
        
        Args:
            fusion_news_embedding: tensor of [batch_size, cdd_size, his_size, transformer_length, repr_dim]
        
        Returns:
            fusion_vectors: tensor of [batch_size, cdd_size, repr_dim]
        """
        fusion_news_embedding = self.tfmInteractor(fusion_news_embedding.view(-1, self.transformer_length, self.repr_dim)).view(self.batch_size, self.cdd_size, self.his_size, self.transformer_length, self.repr_dim)
        fusion_vectors = self._scaled_dp_attention(self.query_itr, fusion_news_embedding, fusion_news_embedding).squeeze(dim=-2)
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

        cdd_repr = self._scaled_dp_attention(self.query_itr, cdd_news_embedding, cdd_news_embedding)
        his_repr = self._scaled_dp_attention(self.query_itr, his_news_embedding, his_news_embedding)

        fusion_vectors = torch.matmul(cdd_repr, his_repr.transpose(-1,-2))


        # fusion_news_embedding = self._fusion(cdd_news_embedding, his_news_embedding)
        # fusion_vectors = self._fusion_transform(fusion_news_embedding)


        score_batch = self._click_predictor(fusion_vectors)
        return score_batch