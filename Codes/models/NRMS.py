import torch
import torch.nn as nn
from .Attention import Attention
from models.base_model import BaseModel
from .Encoders.MHA import MHA_User_Encoder,MHA_Encoder

class NRMS(BaseModel):
    def __init__(self, hparams, vocab, encoder):
        super().__init__(hparams)

        self.name = 'nrms' + encoder.name

        self.title_length = hparams['title_size']
        self.abs_length = hparams['abs_size']

        self.encoder = encoder
        self.user_encoder = MHA_User_Encoder(hparams)
        self.hidden_dim = self.encoder.hidden_dim


    def _click_predictor(self, cdd_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_repr: [batch_size, cdd_size, hidden_dim]
            user_repr: [batch_size, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size], which is normalized click probabilty
        """
        score = torch.matmul(cdd_repr,user_repr.unsqueeze(dim=-1)).squeeze(dim=-1)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, cdd_news_repr = self.encoder(
            cdd_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['cdd_id'].long().to(self.device),
            attn_mask=x['candidate_title_pad'].to(self.device))

        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_repr = self.encoder(
            his_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['his_id'].long().to(self.device),
            attn_mask=x['clicked_title_pad'].to(self.device))

        user_repr = self.user_encoder(his_news_repr)

        score = self._click_predictor(cdd_news_repr, user_repr)

        return score


class NRMS_MultiView(BaseModel):
    def __init__(self, hparams, vocab):
        super().__init__(hparams)

        self.vert_num = hparams['vert_num']
        self.subvert_num = hparams['subvert_num']

        self.encoder = MHA_Encoder(hparams, vocab)
        self.user_encoder = MHA_User_Encoder(hparams)
        self.hidden_dim = self.encoder.hidden_dim

        self.viewQuery = nn.Parameter(torch.randn(1,self.hidden_dim))
        self.vertProject = nn.Linear(self.vert_num, self.hidden_dim)
        self.subvertProject = nn.Linear(self.subvert_num, self.hidden_dim)

        self.name = 'nrms-multiview'

    def _click_predictor(self, cdd_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_repr: [batch_size, cdd_size, hidden_dim]
            user_repr: [batch_size, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size], which is normalized click probabilty
        """
        score = torch.matmul(cdd_repr,user_repr.unsqueeze(dim=-1)).squeeze(dim=-1)
        return score

    def forward_(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_title = x['candidate_title'].long().to(self.device)
        _, cdd_title_repr = self.encoder(cdd_title)

        cdd_abs = x['candidate_abs'].long().to(self.device)
        _, cdd_abs_repr = self.encoder(cdd_abs)

        cdd_vert = x['candidate_vert_onehot'].float().to(self.device)
        cdd_vert_repr = self.vertProject(cdd_vert)
        cdd_subvert = x['candidate_subvert_onehot'].float().to(self.device)
        cdd_subvert_repr = self.subvertProject(cdd_subvert)

        cdd_repr = torch.tanh(torch.stack([cdd_title_repr, cdd_abs_repr, cdd_vert_repr, cdd_subvert_repr], dim=-2))
        cdd_repr = Attention.ScaledDpAttention(self.viewQuery, cdd_repr, cdd_repr).squeeze(dim=-2)

        his_title = x['clicked_title'].long().to(self.device)
        _, his_title_repr = self.encoder(his_title)

        his_abs = x['clicked_abs'].long().to(self.device)
        _, his_abs_repr = self.encoder(his_abs)

        his_vert = x['clicked_vert_onehot'].float().to(self.device)
        his_vert_repr = self.vertProject(his_vert)
        his_subvert = x['clicked_subvert_onehot'].float().to(self.device)
        his_subvert_repr = self.subvertProject(his_subvert)

        his_repr = torch.tanh(torch.stack([his_title_repr, his_abs_repr, his_vert_repr, his_subvert_repr], dim=-2))
        his_repr = Attention.ScaledDpAttention(self.viewQuery, his_repr, his_repr).squeeze(dim=-2)

        user_repr = self.user_encoder(his_repr)
        return self._click_predictor(cdd_title_repr, user_repr)

    def forward(self, x):
        score = self.forward_(x)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score