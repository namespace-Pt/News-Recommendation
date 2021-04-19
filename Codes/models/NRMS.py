import torch
import torch.nn as nn
from .Attention import Attention
from .Encoders.MHA import MHA_Encoder, MHA_User_Encoder

class NRMS(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()

        self.name = 'nrms'

        self.cdd_size = (hparams['npratio'] +
                         1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        self.his_size = hparams['his_size']
        self.title_length = hparams['title_size']
        self.abs_length = hparams['abs_size']

        self.encoder = MHA_Encoder(hparams, vocab)
        self.user_encoder = MHA_User_Encoder(hparams)
        self.hidden_dim = self.encoder.hidden_dim

        self.device = hparams['device']

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

        cdd_title = x['candidate_title'].long().to(self.device)
        _, cdd_title_repr = self.encoder(cdd_title)

        his_title = x['clicked_title'].long().to(self.device)
        _, his_title_repr = self.encoder(his_title)

        user_repr = self.user_encoder(his_title_repr)

        score = self._click_predictor(cdd_title_repr, user_repr)

        return score


class NRMS_MultiView(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()

        self.cdd_size = (hparams['npratio'] +
                         1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        self.his_size = hparams['his_size']
        self.vert_num = hparams['vert_num']
        self.subvert_num = hparams['subvert_num']

        self.encoder = MHA_Encoder(hparams, vocab)
        self.user_encoder = MHA_User_Encoder(hparams)
        self.hidden_dim = self.encoder.hidden_dim

        self.viewQuery = nn.Parameter(torch.randn(1,self.hidden_dim))
        self.vertProject = nn.Linear(self.vert_num, self.hidden_dim)
        self.subvertProject = nn.Linear(self.subvert_num, self.hidden_dim)

        self.device = hparams['device']

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