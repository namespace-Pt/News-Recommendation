import math
import torch
import torch.nn as nn


class FIM_Interactor(nn.Module):
    def __init__(self, level, k=10):
        super().__init__()
        self.name = 'fim'

        if k > 9:
            self.SeqCNN3D = nn.Sequential(
                nn.Conv3d(in_channels=level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
                nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
            )
        else:
            self.SeqCNN3D = nn.Sequential(
                nn.Conv3d(in_channels=level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[1, 3, 3]),
                nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[1, 3, 3])
            )
        nn.init.xavier_normal_(self.SeqCNN3D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN3D[3].weight)


    def forward(self, cdd_news_embedding, his_activated, **kwargs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, final_dim], where final_dim is derived from MaxPooling with no padding
        """
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        # [bs, cs, k, lv, sl, hd]
        his_news_embedding = his_activated.transpose(-2, -3)

        # [batch_size, cdd_size, k, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, his_news_embedding.shape[2], his_news_embedding.shape[3],
                                           his_news_embedding.shape[-2], his_news_embedding.shape[-2]).transpose(1, 2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(cdd_news_embedding.shape[0], cdd_news_embedding.shape[1], -1)
        return fusion_tensor