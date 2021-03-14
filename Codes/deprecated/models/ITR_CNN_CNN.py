import torch
import torch.nn as nn

class GCAModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1

        self.device = torch.device(hparams['device'])
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
        self.SeqCNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
        )
        
        # 64 is derived from SeqCNN
        self.learningToRank = nn.Linear(64, 1)
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
        return attn_output

    def _news_encoder(self,news_batch):
        """ encode batch of news with 1d-CNN
        
        Args:
            news_batch: tensor of [batch_size, *]
        
        Returns:
            news_emebdding: tensor of [batch_size, *, filter_num] 
        """

        news_embedding = self.embedding[news_batch].transpose(-2,-1).view(-1,self.embedding_dim,news_batch.shape[-1])
        news_embedding = self.CNN(news_embedding).transpose(-2,-1).view(news_batch.shape + (self.filter_num,))
        news_embedding = self.ReLU(news_embedding)

        if self.dropout_p > 0:
            news_embedding = self.DropOut(news_embedding)

        return news_embedding
        
    def _fusion(self, cdd_news_embedding, his_news_embedding):
        """ concatenate candidate news title and history news title
        
        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, filter_num] 
            his_news_embedding: tensor of [batch_size, his_size, signal_length, filter_num] 

        Returns:
            fusion_news: tensor of [batch_size, cdd_size, his_size, signal_length, signal_length]
        """

        fusion_matrices = torch.matmul(cdd_news_embedding.unsqueeze(dim=2), his_news_embedding.unsqueeze(dim=1).transpose(-2,-1)).view(self.batch_size * self.cdd_size * self.his_size, 1, self.signal_length, self.signal_length)
        fusion_vectors = self.SeqCNN(fusion_matrices).view(self.batch_size, self.cdd_size, self.his_size, -1)
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
        cdd_news_embedding = self._news_encoder(x['candidate_title'].long().to(self.device))
        his_news_embedding = self._news_encoder(x['clicked_title'].long().to(self.device))

        fusion_vectors = self._fusion(cdd_news_embedding, his_news_embedding)   
        score_batch = self._click_predictor(fusion_vectors)
        return score_batch