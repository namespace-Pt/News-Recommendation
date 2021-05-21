import torch
import math
import torch.nn as nn
from models.Interactors.FIM import FIM_Interactor

class FIMModel(nn.Module):
    def __init__(self,hparams,encoder):
        super().__init__()

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size =hparams['his_size']
        self.batch_size = hparams['batch_size']

        self.signal_length = hparams['title_size']

        self.kernel_size = 3

        self.encoder = encoder
        self.hidden_dim = self.encoder.hidden_dim
        self.level = self.encoder.level
        self.DropOut = self.encoder.DropOut
        self.name = 'fim'

        self.interactor = FIM_Interactor(encoder.level, self.his_size)

        self.device = hparams['device']

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()

        if self.his_size > 9:
            final_dim = int(int(self.his_size / 3) /3) * int(int(self.signal_length / 3) / 3)**2 * 16
        else:
            final_dim = (self.his_size-4) * int(int(self.signal_length / 3) / 3)**2 * 16

        self.learningToRank = nn.Linear(final_dim,1)
        nn.init.xavier_normal_(self.learningToRank.weight)

    def _click_predictor(self,fusion_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, 320]

        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, _ = self.encoder(
            cdd_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['cdd_id'].long().to(self.device))
            # attn_mask=x['candidate_title_pad'].to(self.device))

        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, _ = self.encoder(
            his_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['his_id'].long().to(self.device))
            # attn_mask=x['clicked_title_pad'].to(self.device))

        fusion_tensors = self.interactor(cdd_news_embedding, his_news_embedding.unsqueeze(dim=1))

        score = self._click_predictor(fusion_tensors)
        return score