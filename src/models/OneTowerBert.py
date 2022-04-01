import torch
import torch.nn as nn
from .BaseModel import OneTowerBaseModel
from .modules.encoder import BertCrossEncoder, TFMCrossEncoder



class OneTowerBert(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.encoder = TFMCrossEncoder(manager)

        self.pooler = nn.Linear(manager.plm_dim, 1)
        self.aggregator = nn.Linear(self.his_size, 1)
        nn.init.xavier_normal_(self.pooler.weight)
        nn.init.xavier_normal_(self.aggregator.weight)


    def infer(self, x):
        cdd_token_id = x["cdd_token_id"].to(self.device)    # B, C, L
        his_token_id = x["his_token_id"].to(self.device)    # B, N, L
        cdd_attn_mask = x["cdd_attn_mask"].to(self.device)    # B, C, L
        his_attn_mask = x["his_attn_mask"].to(self.device)    # B, N, L

        B, C, L, N = *cdd_token_id.shape, self.his_size
        cdd_token_id = cdd_token_id.unsqueeze(-2).expand(B, C, N, L)
        his_token_id = his_token_id.unsqueeze(1).expand(B, C, N, L)
        cdd_attn_mask = cdd_attn_mask.unsqueeze(-2).expand(B, C, N, L)
        his_attn_mask = his_attn_mask.unsqueeze(1).expand(B, C, N, L)

        concat_token_id = torch.cat([cdd_token_id, his_token_id], dim=-1)
        concat_attn_mask = torch.cat([cdd_attn_mask, his_attn_mask], dim=-1)

        news_embedding = self.encoder(concat_token_id, concat_attn_mask) # B, C, N, D

        news_his_score = self.pooler(news_embedding).squeeze(-1)    # B, C, N
        logits = self.aggregator(news_his_score).squeeze(-1)
        return logits


    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss