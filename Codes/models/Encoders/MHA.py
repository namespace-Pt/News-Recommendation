import torch
import torch.nn as nn
from ..Attention import Attention


class MHA_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'mha-encoder'

        self.level = 1

        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']
        self.head_num = hparams['head_num']

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=True,freeze=False)

        self.query = nn.Parameter(torch.randn(
            (1, self.query_dim), requires_grad=True))

        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.head_num, dropout=hparams['dropout_p'], kdim=self.embedding_dim, vdim=self.embedding_dim)
        self.queryProject = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.keyProject = nn.Linear(self.hidden_dim, self.query_dim)

        self.DropOut = nn.Dropout(hparams['dropout_p'])


    def forward(self, news_batch, **kwargs):
        """ encode news with built-in multihead attention

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding_pretrained = self.DropOut(self.embedding(news_batch))

        query = self.queryProject(news_embedding_pretrained).view(-1,news_batch.shape[2],self.hidden_dim).transpose(0,1)
        key = news_embedding_pretrained.view(-1,news_batch.shape[2],self.embedding_dim).transpose(0,1)
        value = key

        news_embedding, _ = self.MHA(query, key, value)
        news_embedding = news_embedding.transpose(0,1).view(news_batch.shape + (self.hidden_dim,))

        multi_head_self_attn_key = torch.tanh(self.keyProject(news_embedding))
        news_repr = Attention.ScaledDpAttention(
            self.query, multi_head_self_attn_key, news_embedding).squeeze(dim=-2)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr


class MHA_User_Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.name = 'mha-user-encoder'

        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']
        self.head_num = hparams['head_num']

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.query = nn.Parameter(torch.randn(
            (1, self.query_dim), requires_grad=True))

        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.head_num, dropout=hparams['dropout_p'])
        self.keyProject = nn.Linear(self.hidden_dim, self.query_dim)

        self.DropOut = nn.Dropout(hparams['dropout_p'])


    def forward(self, news_batch, **kwargs):
        """ encode news with built-in multihead attention

        Args:
            news_batch: batch of news tokens, of size [batch_size, cdd_size, his_size, hidden_dim]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        key = news_batch.transpose(0,1)

        news_embedding, _ = self.MHA(key, key, key)
        news_embedding = news_embedding.transpose(0,1)
        multi_head_self_attn_key = torch.tanh(self.keyProject(news_embedding))
        user_repr = Attention.ScaledDpAttention(
            self.query, multi_head_self_attn_key, news_embedding).squeeze(dim=-2)

        return user_repr


class NRMS_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'nrms-encoder'

        self.level = 1

        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']
        self.head_num = hparams['head_num']

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=True,freeze=False)

        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query = nn.Parameter(torch.randn(
            (1, self.query_dim), requires_grad=True))

        # [hn, ed, ed]
        self.queryWeight = nn.Parameter(torch.randn(self.head_num, self.embedding_dim, self.embedding_dim))
        self.queryBias = nn.Parameter(torch.randn(self.head_num, 1, self.embedding_dim))

        self.valueWeight = nn.Parameter(torch.randn(self.head_num, self.embedding_dim, self.value_dim))
        self.valueBias = nn.Parameter(torch.randn(self.head_num, 1, self.value_dim))

        self.keyProject = nn.Linear(self.hidden_dim, self.query_dim)


    def _multi_head_self_attention(self, news_embedding_pretrained):
        """ apply multi-head self attention over input tensor

        Args:
            news_embedding_pretrained: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            additive_attn_repr: tensor of [batch_size, *, repr_dim]
            multi_head_self_attn_value: tensor of [batch_size, *, signal_length, repr_dim]

        """
        # [bs, news_num, 1, sl, ed]
        mha_key = news_embedding_pretrained.unsqueeze(dim=-3)
        # [bs, news_num, head_num, sl, ed]
        mha_query = mha_key.matmul(self.queryWeight) + self.queryBias
        # [bs, news_num, head_num, sl, vd]
        mha_value = Attention.ScaledDpAttention(mha_query, mha_key, mha_key).matmul(self.valueWeight) + self.valueBias

        mha_embedding = mha_value.transpose(-2,-3).reshape(news_embedding_pretrained.shape[0:-1]+(self.hidden_dim,))

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        additive_attn_key = torch.tanh(
            self.keyProject(mha_embedding))
        mha_repr = Attention.ScaledDpAttention(
            self.query, additive_attn_key, mha_embedding).squeeze(dim=-2)
        return mha_embedding, mha_repr

    def forward(self, news_batch, **kwargs):
        """ encode news

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding_pretrained = self.DropOut(self.embedding(news_batch))
        news_embedding, news_repr = self._multi_head_self_attention(
            news_embedding_pretrained)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr
