import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from .BaseModel import TwoTowerBaseModel
from utils.util import pack_results



class TwoTowerGateFormer(TwoTowerBaseModel):
    def __init__(self, manager, newsEncoder, userEncoder, weighter):
        name = "-".join([type(self).__name__, newsEncoder.name, userEncoder.name, f"{manager.enable_gate}_{weighter.name}", str(manager.k)])
        super().__init__(manager, name)

        self.newsEncoder = newsEncoder
        self.userEncoder = userEncoder
        self.weighter = weighter
        self.k = manager.k

        keep_k_modifier = torch.zeros(manager.sequence_length)
        keep_k_modifier[1:self.k + 1] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)


    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
            try:
                gate_mask = x['cdd_gate_mask'].to(self.device)
            except:
                # in case that enable_gate is heuristic
                gate_mask = None

        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
            try:
                gate_mask = x['his_gate_mask'].to(self.device)
            except:
                # in case that enable_gate is heuristic
                gate_mask = None

        token_weight = self.weighter(token_id, attn_mask)
        gated_token_id, gated_attn_mask, gated_token_weight = self._compute_gate(token_id, attn_mask, gate_mask, token_weight)
        news_token_embedding, news_embedding = self.newsEncoder(gated_token_id, gated_attn_mask, gated_token_weight)
        return news_token_embedding, news_embedding



class UserOneTowerGateFormer(TwoTowerBaseModel):
    def __init__(self, manager, encoder, weighter):
        name = "-".join([type(self).__name__, f"{manager.enable_gate}_{weighter.name}", str(manager.k)])
        super().__init__(manager, name)

        self.encoder = encoder
        self.weighter = weighter
        self.k = manager.k

        keep_k_modifier = torch.zeros(manager.sequence_length)
        keep_k_modifier[1:self.k + 1] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)

        self.register_buffer("cls_token_id", torch.tensor(manager.special_token_ids["[CLS]"], dtype=torch.long), persistent=False)
        self.register_buffer("cls_attn_mask", torch.ones(1, dtype=torch.long), persistent=False)
        self.register_buffer("cls_token_weight", torch.ones(1), persistent=False)


    def _encode_news(self, x):
        token_id = x["cdd_token_id"].to(self.device)
        attn_mask = x['cdd_attn_mask'].to(self.device)
        news_token_embedding, news_embedding = self.encoder(token_id, attn_mask)
        return news_token_embedding, news_embedding


    def _encode_user(self, x=None, his_embedding=None, attn_mask=None):
        if x is None:
            # TODO: cache gated token ids, attention masks and weights
            pass
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
            try:
                gate_mask = x['his_gate_mask'].to(self.device)
            except:
                # in case that enable_gate is heuristic
                gate_mask = None
            B = token_id.shape[0]

            try:
                token_weight = self.weighter(token_id, attn_mask)
            except:
                print("fuck weighter")
                raise
            # B, NH, K
            try:
                gated_token_id, gated_attn_mask, gated_token_weight = self._compute_gate(token_id, attn_mask, gate_mask, token_weight)
            except:
                print("fuck gating")
                raise
            gated_token_id = gated_token_id.reshape(B, -1)
            gated_attn_mask = gated_attn_mask.reshape(B, -1)
            gated_token_id = torch.cat([self.cls_token_id.expand(B, 1), gated_token_id], dim=-1)
            gated_attn_mask = torch.cat([self.cls_attn_mask.expand(B, 1), gated_attn_mask], dim=-1)
            if gated_token_weight is not None:
                gated_token_weight = gated_token_weight.reshape(B, -1)
                gated_token_weight = torch.cat([self.cls_token_weight.expand(B, 1), gated_token_weight], dim=-1)

            # B, 1, D
            try:
                user_embedding = self.encoder(gated_token_id, gated_attn_mask, token_weight=gated_token_weight)[1].unsqueeze(-2)
            except:
                print("fuck encoding")
                raise

        return user_embedding


    def forward(self, x):
        _, cdd_news_embedding = self._encode_news(x)
        user_embedding = self._encode_user(x)

        logits = self._compute_logits(cdd_news_embedding, user_embedding)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss


    def infer(self, x):
        """
        one-tower user encoding
        """
        cdd_idx = x["cdd_idx"].to(self.device, non_blocking=True)
        cdd_embedding = self.news_embeddings[cdd_idx]
        user_embedding = self._encode_user(x)

        logits = self._compute_logits(cdd_embedding, user_embedding)
        return logits
