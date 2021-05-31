import os
import logging
import torch
import torch.nn as nn
from ..Attention import Attention
from transformers import AutoModel

class Pipeline_Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.name = 'pipeline-encoder'

        news_repr_path = 'data/tensors/news_repr_{}_{}-[{}].tensor'.format(hparams['scale'],hparams['mode'],hparams['pipeline'])
        news_embedding_path = 'data/tensors/news_embedding_{}_{}-[{}].tensor'.format(hparams['scale'],hparams['mode'],hparams['pipeline'])

        if os.path.exists(news_repr_path) and os.path.exists(news_embedding_path):
            self.news_repr = nn.Embedding.from_pretrained(torch.load(news_repr_path), freeze=True)
            news_embedding = torch.load(news_embedding_path)
            self.news_embedding = nn.Embedding.from_pretrained(news_embedding.view(news_embedding.shape[0],-1), freeze=True)
        else:
            logger = logging.getLogger(__name__)
            logger.warning("No encoded news at '{}', please encode news first!".format(news_embedding_path))
            raise ValueError

        self.level = news_embedding.shape[-2]
        self.hidden_dim = news_embedding.shape[-1]
        self.DropOut = nn.Dropout(hparams['dropout_p'])

    def forward(self,news_batch,**kwargs):
        """ encode news by lookup table

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_repr = self.DropOut(self.news_repr(kwargs['news_id']))
        news_embedding = self.DropOut(self.news_embedding(kwargs['news_id']).view(news_batch.shape + (self.level, self.hidden_dim)))
        return news_embedding, news_repr


class CNN_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'cnn-encoder'

        self.dropout_p = hparams['dropout_p']

        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=True,freeze=False)

        # project preference query to vector of hidden_dim
        self.wordQueryProject = nn.Linear(self.hidden_dim, self.hidden_dim)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])


    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)
        news_embedding = self.RELU(self.LayerNorm(self.CNN(
            news_embedding_pretrained).transpose(-2, -1))).view(news_batch.shape + (self.hidden_dim,))

        news_repr = Attention.ScaledDpAttention(self.query_words, self.Tanh(self.wordQueryProject(news_embedding)), news_embedding).squeeze(dim=-2)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr


class RNN_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'bert-encoder'

        self.level = 2

        self.embedding_dim = hparams['embedding_dim']
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=True,freeze=False)

        # dimension for the final output embedding/representation
        self.hidden_dim = 200

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True,bidirectional=True)

    def forward(self, news_batch, **kwargs):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """

        # conpress news_num into batch_size
        news_embedding_pretrained = self.embedding(news_batch).view(-1, news_batch.shape[-1], self.embedding_dim)
        news_embedding,output = self.lstm(news_embedding_pretrained)
        news_repr = torch.mean(output[0],dim=0).view(news_batch.shape[0],news_batch.shape[1],self.hidden_dim)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr

class Encoder_Wrapper(nn.Module):
    def __init__(self, hparams, encoder):
        super().__init__()
        self.encoder = encoder
        self.name = 'pipeline-'+encoder.name

        self.hidden_dim = encoder.hidden_dim
        self.level = encoder.level

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']


    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        news = x['candidate_title'].long()
        news_embedding, news_repr = self.encoder(
            news,
            news_id=x['cdd_id'].long(),
            attn_mask=x['candidate_title_pad'])

        return news_embedding, news_repr

class Bert_Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.name = 'bert-encoder'

        self.level = hparams['level']

        # dimension for the final output embedding/representation
        self.hidden_dim = 768

        self.bert = AutoModel.from_pretrained(
            hparams['bert'],
            # output hidden embedding of each transformer layer
            output_hidden_states=True
        )

    def forward(self, news_batch, **kwargs):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        output = self.bert(news_batch.view(-1, news_batch.shape[2]), attention_mask=kwargs['attn_mask'].view(-1, news_batch.shape[2]))
        # stack the last level dimensions to form multi-level embedding
        news_embedding = torch.stack(output['hidden_states'][-self.level:],dim=-2)
        # use the hidden state of the last layer of [CLS] as news representation
        news_repr = news_embedding[:,0,-1,:].view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr