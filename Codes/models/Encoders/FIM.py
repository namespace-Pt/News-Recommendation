import torch
import torch.nn as nn
from ..Attention import Attention

class FIM_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'fim-encoder'

        self.kernel_size = 3
        self.level = 3

        # concatenate category embedding and subcategory embedding
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=True,freeze=False)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.device = hparams['device']
        self.attrs = hparams['attrs']

        nn.init.xavier_normal_(self.CNN_d1.weight)
        nn.init.xavier_normal_(self.CNN_d2.weight)
        nn.init.xavier_normal_(self.CNN_d3.weight)


    def _HDC(self, news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3

        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros(
            (news_embedding_set.shape[0], news_embedding_set.shape[1], self.level, self.hidden_dim), device=news_embedding_set.device)

        news_embedding_set = news_embedding_set.transpose(-2,-1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_set)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)

        news_embedding_d3 = self.CNN_d3(news_embedding_set)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,:,2,:] = self.ReLU(news_embedding_d3)

        return news_embedding_dilations

    def forward(self, news_batch, **kwargs):
        """ encode set of news to news representation

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding = self.DropOut(
            self.embedding(news_batch)).view(-1, news_batch.shape[2], self.embedding_dim)
        news_embedding = self._HDC(news_embedding).view(
            news_batch.shape + (self.level, self.hidden_dim))
        news_embedding_attn = Attention.ScaledDpAttention(
            self.query_levels, news_embedding, news_embedding).squeeze(dim=-2)
        news_repr = Attention.ScaledDpAttention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(
            dim=-2).view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)

        return news_embedding, news_repr