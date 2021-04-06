import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class FIM_Interactor(nn.Module):
    def __init__(self):
        super().__init__()
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=self.level, out_channels=32,
                      kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16,
                      kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )

    def forward(self, cdd_news_embedding, his_news_embedding):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, *], where * is derived from MaxPooling with no padding
        """
        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        his_news_embedding = his_news_embedding.transpose(-2, -3)

        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, his_news_embedding.shape[2], self.level,
                                           self.signal_length, self.signal_length).transpose(1, 2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(self.batch_size, self.cdd_size, -1)
        return fusion_tensor

class KNRM_Interactor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cdd_news_embedding, his_news_embedding):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, *], where * is derived from MaxPooling with no padding
        """
        cdd_news_embedding = F.normalize(cdd_news_embedding.transpose(-2, -3), dim=-1)
        his_news_embedding = F.normalize(his_news_embedding.transpose(-2, -3), dim=-1)

        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        kernel


    def _fusion(self, cdd_news_batch, his_news_batch):
        """ fuse batch of candidate news and history news into batch of |candidate|*|history| interaction matrixs, according to cosine similarity

        Args:
            cdd_news_batch: tensor of [batch_size, cdd_size, signal_length]
            his_news_batch: tensor of [batch_size, his_size, signal_length]

        Returns:
            fusion_matrixs: tensor of [batch_size, cdd_size, his_size, signal_length, signal_length]
        """
        # [bs, cs, 1, sl, ed]
        cdd_news_embedding = F.normalize(self.embedding(cdd_news_batch).unsqueeze(dim=2), dim=-1)
        # [bs, 1, hs, ed, sl]
        his_news_embedding = F.normalize(self.embedding(his_news_batch).unsqueeze(dim=1), dim=-1).transpose(-1,-2)

        # transform cosine similarity calculation into normalized matrix production
        fusion_matrices = torch.matmul(cdd_news_embedding, his_news_embedding).unsqueeze(dim=-1)

        return fusion_matrices