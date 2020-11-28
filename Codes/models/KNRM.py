import torch
import torch.nn as nn
class KNRMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.mus = torch.arange(-0.9,1.1,0.1,device=self.device)
        self.sigmas = torch.tensor([0.1]*10 + [0.001], device=self.device)
        self.kernel_num = 11

    def _kernel_pooling(self,matrix):
        """
            apply kernel pooling on matrix, in order to get the relatedness from many levels
        
        Args:
            matrix: tensor of [batch_size, rows, columns]
        
        Returns:
            pooling_vector: tensor of [batch_size, kernel_num]
        """
        pooling_matrix = torch.zeros(matrix.shape[0],matrix.shape[1],self.kernel_num,device=self.device)
        
        for k in range(self.kernel_num):
            pooling_matrix[:,:,k] = torch.sum(torch.exp(-(matrix - self.mus[k])**2 / (2*self.sigmas[k]**2)),dim=2)
        
        pooling_vector = torch.sum(torch.log(pooling_matrix),dim=1)
        
        return pooling_vector