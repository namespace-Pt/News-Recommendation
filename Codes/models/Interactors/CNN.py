import torch
import math
import torch.nn as nn

class CNN_Interator(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.name = '2dcnn'

        # FIXED
        self.hidden_dim = 64

        self.SeqCNN2D = nn.Sequential(
            nn.Conv2d(in_channels=k, out_channels=32, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3]),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3])
        )
        nn.init.xavier_normal_(self.SeqCNN2D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN2D[3].weight)

    def forward(self, cdd_news_embedding, his_activated, **kwargs):
        """ construct fusion tensor between candidate news repr and history news repr with 2DCNN

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, final_dim]
        """
        # [batch_size, cdd_size, k, level, signal_length, hidden_dim]
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        his_news_embedding = his_activated.transpose(-2, -3)

        # [batch_size, cdd_size, k, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        if fusion_tensor.size(3) > 1:
            fusion_tensor = torch.sum(fusion_tensor,dim=3)
        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, his_news_embedding.shape[2], his_news_embedding.shape[-2], his_news_embedding.shape[-2])
        fusion_tensor = self.SeqCNN2D(fusion_tensor).view(cdd_news_embedding.shape[0], cdd_news_embedding.shape[1], -1)

        return fusion_tensor
