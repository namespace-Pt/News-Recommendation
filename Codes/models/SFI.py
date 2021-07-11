import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attention import Attention
from models.base_model import BaseModel

class SFI(BaseModel):
    def __init__(self, hparams, encoder, interactor):
        super().__init__(hparams)

        self.title_size = hparams['title_size']
        self.k = hparams['k']

        self.encoder = encoder
        self.level = encoder.level
        self.hidden_dim = encoder.hidden_dim

        self.interactor = interactor
        if interactor.name == 'fim':
            if self.k > 9:
                final_dim = int(int(self.k / 3) /3) * int(int(self.title_size / 3) / 3)**2 * 16
            else:
                final_dim = (self.k-4) * int(int(self.title_size / 3) / 3)**2 * 16

        elif interactor.name == '2dcnn':
            final_dim = 16 * int(int(self.title_size / 3) / 3)**2
        else:
            final_dim = interactor.hidden_dim

        self.learningToRank = nn.Sequential(
            nn.Linear(final_dim, int(final_dim/2)),
            nn.ReLU(),
            nn.Linear(int(final_dim/2),1)
        )
        self.name = '-'.join(['sfi', self.encoder.name, self.interactor.name])

        self.selectionProject = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        if hparams['threshold'] != -float('inf'):
            threshold = torch.tensor([hparams['threshold']])
            self.register_buffer('threshold', threshold)

        for param in self.selectionProject:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
        for param in self.learningToRank:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)


    def _history_filter(self, cdd_repr, his_repr, his_embedding):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, hidden_dim]
            his_repr: tensor of [batch_size, his_size, hidden_dim]
            his_embedding: tensor of [batch_size, his_size, title_size, level, hidden_dim]

        Returns:
            his_activated: tensor of [batch_size, cdd_size, k, title_size, hidden_dim]
            his_focus: tensor of [batch_size, cdd_size, k, his_size]
            pos_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
            neg_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
        """

        # [bs, cs, hs]
        # t1 = time.time()
        cdd_repr = F.normalize(self.selectionProject(cdd_repr),dim=-1)
        his_repr = F.normalize(self.selectionProject(his_repr),dim=-1)
        attn_weights = cdd_repr.matmul(his_repr.transpose(-1, -2))

        if self.k == self.his_size:

            his_activated = his_embedding.unsqueeze(dim=1)
            attn_weights_index = None

        else:
            # t2 = time.time()
            # attn_weights = attn_weights.masked_fill(his_mask.transpose(-1, -2), -float("inf"))
            attn_weights, attn_weights_index = attn_weights.topk(dim=-1, k=self.k)

            # print(attn_weights, attn_weights_index)
            # t3 = time.time()

            # [bs, cs, k, sl, level, fn]
            his_activated = his_embedding.unsqueeze(dim=1).expand(self.batch_size, self.cdd_size, self.his_size, self.title_size, self.level, self.hidden_dim).gather(dim=2, index=attn_weights_index.view(self.batch_size,self.cdd_size,self.k,1,1,1).expand(self.batch_size,self.cdd_size,self.k,self.title_size,self.level,self.hidden_dim))

            # t4 = time.time()

        if hasattr(self,'threshold'):
            his_activated = his_activated * (attn_weights.masked_fill(attn_weights<self.threshold, 0).view(self.batch_size, self.cdd_size, self.k, 1, 1, 1))

        # t6 = time.time()
        # print("product time:{}, sort time:{}, scatter time:{}, activate time:{}, mask time:{}".format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))

        output = (his_activated, attn_weights_index)
        return output

    def _click_predictor(self, fusion_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, *]

        Returns:
            score: tensor of [batch_size, cdd_size], which is normalized click probabilty
        """
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
        return score

    def _forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, cdd_news_repr = self.encoder(
            cdd_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['cdd_id'].long().to(self.device))
            # attn_mask=x['candidate_title_pad'].to(self.device))

        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_repr = self.encoder(
            his_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['his_id'].long().to(self.device))
            # attn_mask=x['clicked_title_pad'].to(self.device))

        # t2 = time.time()

        output = self._history_filter(
            cdd_news_repr, his_news_repr, his_news_embedding)

        # t3 = time.time()

        if self.interactor.name == 'knrm':
            cdd_pad = x['candidate_title_pad'].float().to(self.device).view(self.batch_size, self.cdd_size, 1, 1, -1, 1)
            if output[1] is not None:
                his_pad = x['clicked_title_pad'].float().to(self.device).unsqueeze(dim=1).expand(self.batch_size, self.cdd_size, self.his_size, self.title_size).gather(dim=2, index=output[1].unsqueeze(dim=-1).expand(self.batch_size, self.cdd_size, self.k, self.title_size)).view(self.batch_size, self.cdd_size, self.k, 1, 1, self.title_size, 1)
            else:
                his_pad = x['clicked_title_pad'].float().to(self.device).view(self.batch_size, 1, self.k, 1, 1, self.title_size, 1).expand(self.batch_size, self.cdd_size, self.k, 1, 1, self.title_size, 1)

            fusion_tensors = self.interactor(cdd_news_embedding, output[0], cdd_pad=cdd_pad, his_pad=his_pad)

        else:
            fusion_tensors = self.interactor(cdd_news_embedding, output[0])

        # t4 = time.time()
        # print(fusion_tensors.shape)
        # print("encoding time:{} selection time:{} interacting time:{}".format(t2-t1, t3-t2, t4-t3))

        return self._click_predictor(fusion_tensors)

    def forward(self, x):
        score = self._forward(x)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score


class SFI_unified(BaseModel):
    def __init__(self, hparams, encoder, interactor):
        super().__init__(hparams)

        self.title_size = hparams['title_size']
        self.k = hparams['k']

        self.encoder = encoder
        self.level = encoder.level
        self.hidden_dim = encoder.hidden_dim

        self.interactor = interactor
        if interactor.name == 'fim':
            if self.k > 9:
                final_dim = int(int(self.k / 3) /3) * int(int(self.title_size / 3) / 3)**2 * 16
            else:
                final_dim = (self.k-4) * int(int(self.title_size / 3) / 3)**2 * 16

        elif interactor.name == '2dcnn':
            final_dim = 16 * int(int(self.title_size / 3) / 3)**2
        else:
            final_dim = interactor.hidden_dim

        final_dim += self.his_size

        if final_dim > 100:
            self.learningToRank = nn.Sequential(
                nn.Linear(final_dim, 50),
                nn.ReLU(),
                nn.Linear(50,1)
            )
        else:
            self.learningToRank = nn.Sequential(
                nn.Linear(final_dim, 1)
            )

        self.selectionProject = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.name = '-'.join(['sfi-coarse', encoder.name, interactor.name])

        if hparams['threshold'] != -float('inf'):
            threshold = torch.tensor([hparams['threshold']])
            self.register_buffer('threshold', threshold)

        for param in self.selectionProject:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
        for param in self.learningToRank:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)


    def _history_filter(self, cdd_repr, his_repr, his_embedding):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, hidden_dim]
            his_repr: tensor of [batch_size, his_size, hidden_dim]
            his_embedding: tensor of [batch_size, his_size, title_size, level, hidden_dim]

        Returns:
            his_activated: tensor of [batch_size, cdd_size, k, title_size, hidden_dim]
            his_focus: tensor of [batch_size, cdd_size, k, his_size]
            pos_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
            neg_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
        """

        # [bs, cs, hs]
        # t1 = time.time()
        cdd_repr = F.normalize(self.selectionProject(cdd_repr),dim=-1)
        his_repr = F.normalize(self.selectionProject(his_repr),dim=-1)
        attn_weights = cdd_repr.matmul(his_repr.transpose(-1, -2))

        if self.k == self.his_size:

            his_activated = his_embedding.unsqueeze(dim=1)
            attn_weights_index = None

        else:
            # t2 = time.time()
            attn_weights, attn_weights_index = attn_weights.topk(dim=-1, k=self.k)

            # print(attn_weights, attn_weights_index)
            # t3 = time.time()

            # [bs, cs, k, sl, level, fn]
            his_activated = his_embedding.unsqueeze(dim=1).expand(self.batch_size, self.cdd_size, self.his_size, self.title_size, self.level, self.hidden_dim).gather(dim=2, index=attn_weights_index.view(self.batch_size,self.cdd_size,self.k,1,1,1).expand(self.batch_size,self.cdd_size,self.k,self.title_size,self.level,self.hidden_dim))

            # t4 = time.time()

        if hasattr(self,'threshold'):
            his_activated = his_activated * (attn_weights.masked_fill(attn_weights<self.threshold, 0).view(self.batch_size, self.cdd_size, self.k, 1, 1, 1))

        # t6 = time.time()
        # print("product time:{}, sort time:{}, scatter time:{}, activate time:{}, mask time:{}".format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))

        output = (his_activated, attn_weights_index)
        return output

    def _click_predictor(self, itr_tensors, repr_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, *]

        Returns:
            score: tensor of [batch_size, cdd_size], which is normalized click probabilty
        """
        # score_itr = self.learningToRank(itr_tensors).squeeze(dim=-1)
        # score_repr = self.coarse_combine(repr_tensors).squeeze(dim=-1)
        score = self.learningToRank(torch.cat([itr_tensors, repr_tensors], dim=-1)).squeeze(dim=-1)
        # score = self.a * score_itr + (1-self.a) * score_repr
        return score

    def _forward(self, x):
        # t1 = time.time()

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

        # t2 = time.time()

        output = self._history_filter(
            cdd_news_repr, his_news_repr, his_news_embedding)

        # t3 = time.time()

        if self.interactor.name == 'knrm':
            cdd_pad = x['candidate_title_pad'].float().to(self.device).view(self.batch_size, self.cdd_size, 1, 1, -1, 1)
            if output[1] is not None:
                his_pad = x['clicked_title_pad'].float().to(self.device).unsqueeze(dim=1).expand(self.batch_size, self.cdd_size, self.his_size, self.title_size).gather(dim=2, index=output[1].unsqueeze(dim=-1).expand(self.batch_size, self.cdd_size, self.k, self.title_size)).view(self.batch_size, self.cdd_size, self.k, 1, 1, self.title_size, 1)
            else:
                his_pad = x['clicked_title_pad'].float().to(self.device).view(self.batch_size, 1, self.k, 1, 1, self.title_size, 1).expand(self.batch_size, self.cdd_size, self.k, 1, 1, self.title_size, 1)

            itr_tensors = self.interactor(cdd_news_embedding, output[0], cdd_pad=cdd_pad, his_pad=his_pad)

        else:
            itr_tensors = self.interactor(cdd_news_embedding, output[0])

        # t4 = time.time()
        # print(fusion_tensors.shape)
        # print("encoding time:{} selection time:{} interacting time:{}".format(t2-t1, t3-t2, t4-t3))

        repr_tensors = cdd_news_repr.unsqueeze(dim=2).matmul(his_news_repr.unsqueeze(dim=1).transpose(-2,-1)).squeeze(dim=-2)

        return self._click_predictor(itr_tensors, repr_tensors)

    def forward(self, x):
        score = self._forward(x)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score

class SFI_MultiView(BaseModel):
    def __init__(self, hparams, encoder, interactor):
        super().__init__(hparams)

        self.title_size = hparams['title_size'] + 2
        self.abs_size = hparams['abs_size']

        self.k = hparams['k']

        if(encoder.name != 'fim' or interactor.name != 'fim'):
            logging.error("please use FIM encoder and FIM interactor")

        self.encoder = encoder
        self.level = encoder.level
        self.hidden_dim = encoder.hidden_dim

        self.interactor = interactor
        if self.k > 9:
            title_dim = int(int(self.k / 3) /3) * int(int(self.title_size / 3) / 3)**2 * 16
            abs_dim = int(int(self.k / 3) /3) * int(int(self.abs_size / 3) / 3)**2 * 16
        else:
            title_dim = (self.k - 4) * int(int(self.title_size / 3) / 3)**2 * 16
            abs_dim = (self.k - 4)* int(int(self.abs_size / 3) / 3)**2 * 16

        # final_dim += self.his_size

        self.view_dim = 200
        self.title2view = nn.Linear(title_dim,200)
        self.abs2view = nn.Linear(abs_dim,200)
        self.viewQuery = nn.Parameter(torch.randn(1,self.view_dim))

        self.learningToRank = nn.Sequential(
            nn.Linear(self.view_dim, 50),
            nn.ReLU(),
            nn.Linear(50,1)
        )

        self.selectionProject = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.name = '-'.join(['sfi-multiview', encoder.name, interactor.name])

        if hparams['threshold'] != -float('inf'):
            threshold = torch.tensor([hparams['threshold']])
            self.register_buffer('threshold', threshold)

        for param in self.selectionProject:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
        for param in self.learningToRank:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)

        nn.init.xavier_normal_(self.title2view.weight)
        nn.init.xavier_normal_(self.abs2view.weight)


    def _history_filter(self, cdd_repr, his_repr, his_embedding):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, hidden_dim]
            his_repr: tensor of [batch_size, his_size, hidden_dim]
            his_embedding: tensor of [batch_size, his_size, signal_length, level, hidden_dim]

        Returns:
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]
            his_focus: tensor of [batch_size, cdd_size, k, his_size]
            pos_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
            neg_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
        """

        # [bs, cs, hs]
        # t1 = time.time()
        cdd_repr = F.normalize(self.selectionProject(cdd_repr),dim=-1)
        his_repr = F.normalize(self.selectionProject(his_repr),dim=-1)
        signal_length = his_embedding.size(-3)
        attn_weights = cdd_repr.matmul(his_repr.transpose(-1, -2))

        if self.k == self.his_size:

            his_activated = his_embedding.unsqueeze(dim=1)
            attn_weights_index = None

        else:
            # t2 = time.time()
            attn_weights, attn_weights_index = attn_weights.topk(dim=-1, k=self.k)

            # print(attn_weights, attn_weights_index)
            # t3 = time.time()

            # [bs, cs, k, sl, level, fn]
            his_activated = his_embedding.unsqueeze(dim=1).expand(self.batch_size, self.cdd_size, self.his_size, signal_length, self.level, self.hidden_dim).gather(dim=2, index=attn_weights_index.view(self.batch_size,self.cdd_size,self.k,1,1,1).expand(self.batch_size,self.cdd_size,self.k,signal_length,self.level,self.hidden_dim))

            # t4 = time.time()

        if hasattr(self,'threshold'):
            his_activated = his_activated * (attn_weights.masked_fill(attn_weights<self.threshold, 0).view(self.batch_size, self.cdd_size, self.k, 1, 1, 1))

        # t6 = time.time()
        # print("product time:{}, sort time:{}, scatter time:{}, activate time:{}, mask time:{}".format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))

        output = (his_activated, attn_weights_index)
        return output

    def _click_predictor(self, fusion_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, *]

        Returns:
            score: tensor of [batch_size, cdd_size], which is normalized click probabilty
        """
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
        return score

    def _forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        # FIXME, according to FIM, the category is concatenated into title before padding
        cdd_title = torch.cat([x['candidate_title'], x['candidate_vert'], x['candidate_subvert']], dim=-1).long().to(self.device)
        cdd_title_embedding, cdd_title_repr = self.encoder(
            cdd_title,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['cdd_id'].long().to(self.device))
            # attn_mask=x['candidate_title_pad'].to(self.device))

        his_title = torch.cat([x['clicked_title'], x['clicked_vert'], x['clicked_subvert']], dim=-1).long().to(self.device)
        his_title_embedding, his_title_repr = self.encoder(
            his_title,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['his_id'].long().to(self.device))
            # attn_mask=x['clicked_title_pad'].to(self.device))

        cdd_abs = x['candidate_abs'].long().to(self.device)
        cdd_abs_embedding, cdd_abs_repr = self.encoder(
            cdd_abs,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['cdd_id'].long().to(self.device))
            # attn_mask=x['candidate_abs_pad'].to(self.device))

        his_abs = x['clicked_abs'].long().to(self.device)
        his_abs_embedding, his_abs_repr = self.encoder(
            his_abs,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['his_id'].long().to(self.device))
            # attn_mask=x['clicked_abs_pad'].to(self.device))

        output_title = self._history_filter(
            cdd_title_repr, his_title_repr, his_title_embedding)

        output_abs = self._history_filter(
            cdd_abs_repr, his_abs_repr, his_abs_embedding)

        fusion_tensors_title = self.interactor(cdd_title_embedding, output_title[0])
        fusion_tensors_abs = self.interactor(cdd_abs_embedding, output_abs[0])

        # [bs, cs, 2, hd]
        fusion_tensors = torch.tanh(torch.stack([self.title2view(fusion_tensors_title), self.abs2view(fusion_tensors_abs)], dim=-2))
        fusion_tensor = Attention.ScaledDpAttention(self.viewQuery, fusion_tensors, fusion_tensors).squeeze(dim=-2)
        return self._click_predictor(fusion_tensor)

    def forward(self, x):
        score = self._forward(x)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score