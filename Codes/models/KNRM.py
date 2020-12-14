import torch
import torch.nn as nn
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
        
        self.mus = torch.arange(-0.9,1.1,0.1,device=self.device)
        self.kernel_num = len(self.mus)
        self.sigmas = torch.tensor([0.1]*(self.kernel_num - 1) + [0.001], device=self.device)

        self.embedding = vocab.vectors.clone().detach().requires_grad_(True).to(self.device)
        self.CosSim = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.learningToRank = nn.Linear(self.his_size * self.kernel_num, 1)

    def _kernel_pooling(self,matrixs):
        """
            apply kernel pooling on matrix, in order to get the relatedness from many levels
        
        Args:
            matrix: tensor of [batch_size, rows, columns]
        
        Returns:
            pooling_vectors: tensor of [batch_size, kernel_num]
        """
        pooling_matrixs = torch.zeros(matrixs.shape[0],matrixs.shape[1],self.kernel_num,device=self.device)
        
        for k in range(self.kernel_num):
            pooling_matrixs[:,:,k] = torch.sum(torch.exp(-(matrixs - self.mus[k])**2 / (2*self.sigmas[k]**2)),dim=2)
        
        pooling_vectors = torch.sum(torch.log(torch.clamp(pooling_matrixs,min=1e-10)),dim=1)
        return pooling_vectors
    
    def _fusion(self, cdd_news_batch, his_news_batch):
        """ fuse batch of candidate news and history news into batch of |candidate|*|history| interaction matrixs, according to cosine similarity

        Args:
            cdd_news_batch: tensor of [batch_size, cdd_size, signal_length]
            his_news_batch: tensor of [batch_size, his_size, signal_length]
        
        Returns:
            fusion_matrixs: tensor of [batch_size, cdd_size, his_size, signal_length, signal_length]
        """
        cdd_news_embedding = self.embedding[cdd_news_batch]
        his_news_embedding = self.embedding[his_news_batch]
        
        fusion_matrixes = torch.zeros((self.batch_size, self.cdd_size, self.his_size, self.signal_length, self.signal_length), device=self.device)
        for i in range(self.cdd_size):
            for j in range(self.signal_length):
                fusion_matrixes[:,i,:,j,:] = self.CosSim(cdd_news_embedding[:,i,j,:].unsqueeze(1).unsqueeze(2), his_news_embedding)

        return fusion_matrixes

    def _click_predictor(self, pooling_vectors):
        """ learning to rank
        Args: 
            pooling_vecors: tensor of [batch_size, cdd_size, his_size * kernel_num]
        
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
        fusion_matrixes = self._fusion(x['candidate_title'].long().to(self.device),x['clicked_title'].long().to(self.device))
        pooling_vectors = self._kernel_pooling(fusion_matrixes.view(-1,self.signal_length,self.signal_length)).view(self.batch_size,self.cdd_size,-1)
        score = self._click_predictor(pooling_vectors)

        return score