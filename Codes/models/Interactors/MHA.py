import torch
import torch.nn as nn
from ..Attention import Attention


class MHA_Interactor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.name = 'mha-encoder'

        self.level = 1

        self.embedding_dim = hidden_dim
        self.value_dim = 16
        self.query_dim = 200
        self.head_num = 16

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.query_words = nn.Parameter(torch.randn((1, self.hidden_dim), requires_grad=True))
        self.query_news = nn.Parameter(torch.randn((1, self.hidden_dim), requires_grad=True))

        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.head_num, dropout=0.2, kdim=self.embedding_dim, vdim=self.embedding_dim)
        self.queryProject = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.keyProject = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, cdd_news_embedding, his_activated, **kwargs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, final_dim], where final_dim is derived from MaxPooling with no padding
        """

        # [batch_size, cdd_size, k, signal_length, hidden_dim]
        shape = torch.Size((his_activated.size(0), his_activated.size(1), his_activated.size(2), 2*his_activated.size(3), self.embedding_dim))
        if cdd_news_embedding.size(-2) > 1:
            cdd_news_embedding = torch.sum(cdd_news_embedding, dim=-2).unsqueeze(dim=2).expand((his_activated.size(0), his_activated.size(1), his_activated.size(2), his_activated.size(3), self.embedding_dim))
            his_news_embedding = torch.sum(his_activated, dim=-2)

        else:
            cdd_news_embedding = cdd_news_embedding.squeeze(dim=-2).unsqueeze(dim=2).expand((his_activated.size(0), his_activated.size(1), his_activated.size(2), his_activated.size(3), self.embedding_dim))
            his_news_embedding = his_activated.squeeze(dim=-2)

        fusion_tensors = torch.cat([cdd_news_embedding, his_news_embedding], dim=-2).view(-1, shape[-2], self.embedding_dim)
        query = self.queryProject(fusion_tensors).transpose(0,1)
        key = fusion_tensors.transpose(0,1)
        value = key

        fusion_tensors,_ = self.MHA(query, key, value)
        fusion_tensors = fusion_tensors.transpose(0,1).view(shape[0:-1] + (self.hidden_dim,))

        key = torch.tanh(self.keyProject(fusion_tensors))
        fusion_tensor = Attention.ScaledDpAttention(
            self.query_words, key, fusion_tensors).squeeze(dim=-2)

        fusion_tensor = Attention.ScaledDpAttention(
            self.query_news, fusion_tensor, fusion_tensor).squeeze(dim=-2)

        return fusion_tensor