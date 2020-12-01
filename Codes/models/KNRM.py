import torch
import torch.nn as nn
class KNRMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.mus = torch.arange(-0.9,1.1,0.1,device=self.device)
        self.kernel_num = len(self.mus)
        self.sigmas = torch.tensor([0.1]*(self.kernel_num - 1) + [0.001], device=self.device)

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