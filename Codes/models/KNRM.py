import torch
import torch.nn as nn
import torch.nn.functional as F

class KNRMModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']
        self.device = hparams['device']
        self.metrics = hparams['metrics']

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']
        self.embedding_dim = hparams['embedding_dim']
        
        mus = torch.arange(-0.9,1.1,0.1,device=self.device)
        self.kernel_num = len(mus)
        self.mus = mus.view(1,1,1,1,1,-1)
        self.sigmas = torch.tensor([0.1]*(self.kernel_num - 1) + [0.001], device=self.device).view(1,1,1,1,1,-1)
        if hparams['train_embedding']:
            self.embedding = vocab.vectors.clone().detach().requires_grad_(True).to(self.device)
        else:
            self.embedding = vocab.vectors.to(self.device)
        
        self.query = nn.Parameter(torch.randn((1,self.kernel_num), requires_grad=True))

        self.softmax = nn.Softmax(dim=-1)
        self.learningToRank = nn.Linear(self.kernel_num, 1)

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
    
    def _fusion(self, cdd_news_batch, his_news_batch):
        """ fuse batch of candidate news and history news into batch of |candidate|*|history| interaction matrixs, according to cosine similarity

        Args:
            cdd_news_batch: tensor of [batch_size, cdd_size, signal_length]
            his_news_batch: tensor of [batch_size, his_size, signal_length]
        
        Returns:
            fusion_matrixs: tensor of [batch_size, cdd_size, his_size, signal_length, signal_length]
        """
        # [bs, cs, 1, sl, ed]
        cdd_news_embedding = F.normalize(self.embedding[cdd_news_batch].unsqueeze(dim=2), dim=-1)
        # [bs, 1, hs, ed, sl]
        his_news_embedding = F.normalize(self.embedding[his_news_batch].unsqueeze(dim=1), dim=-1).transpose(-1,-2)

        # transform cosine similarity calculation into normalized matrix production
        fusion_matrices = torch.matmul(cdd_news_embedding, his_news_embedding).unsqueeze(dim=-1)

        return fusion_matrices

    def _kernel_pooling(self, matrices, mask_cdd, mask_his):
        """
            apply kernel pooling on matrix, in order to get the relatedness from many levels
        
        Args:
            matrices: tensor of [batch_size, cdd_size, his_size, signal_length, signal_length, 1]
            mask_cdd: tensor of [batch_size, cdd_size, 1, signal_length, 1]
            mask_his: tensor of [batch_size, 1, his_size, 1, signal_length, 1]
        
        Returns:
            pooling_vectors: tensor of [batch_size, cdd_size, his_size, kernel_num]
        """

        pooling_matrices = torch.exp(-((matrices - self.mus) ** 2) / (2 * self.sigmas ** 2)) * mask_his
        pooling_sum = torch.sum(pooling_matrices, dim=-2)
        
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * mask_cdd * 0.01
        pooling_vectors = torch.sum(log_pooling_sum, dim=-2)
        return pooling_vectors

    def _his_combine(self, pooling_vectors):
        """ pack all history interaction result into one single vector

        Args:
            pooling_vectors: tensor of [batch_size, cdd_size, his_size, kernel_num]
        Returns:
            pooling_vector: tensor of [batch_size, cdd_size, kernel_num]
        """
        pooling_vector = self._scaled_dp_attention(self.query, pooling_vectors, pooling_vectors)
        return pooling_vector

    def _click_predictor(self, pooling_vectors):
        """ learning to rank
        Args: 
            pooling_vecors: tensor of [batch_size, cdd_size, kernel_num]
        
        Returns:
            scpre: tensor of [batch_size, cdd_size, his_size]
        """
        score = self.learningToRank(pooling_vectors)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score.squeeze()

    def forward(self, x):
        fusion_matrices = self._fusion(x['candidate_title'].long().to(self.device),x['clicked_title'].long().to(self.device))
        pooling_vectors = self._kernel_pooling(fusion_matrices, x['candidate_title_pad'].float().to(self.device).view(self.batch_size, self.cdd_size, 1, self.signal_length, 1), x['clicked_title_pad'].float().to(self.device).view(self.batch_size, 1, self.his_size, 1, self.signal_length, 1))
        pooling_vector = self._his_combine(pooling_vectors)
        # score = self._click_predictor(pooling_vectors.view(self.batch_size, self.cdd_size, -1))
        score = self._click_predictor(pooling_vector)
        return score