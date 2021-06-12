import os
import logging
import torch
import torch.nn as nn

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