import torch
import torch.nn as nn
from transformers import AutoModel
from .attention import TFMLayer


class BaseWeighter(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = type(self).__name__[:-8]

        self.weightPooler = nn.Sequential(
            nn.Linear(manager.gate_hidden_dim, manager.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(manager.dropout_p),
            nn.Linear(manager.gate_hidden_dim, 1)
        )


    def _compute_weight(self, embeddings):
        weights = self.weightPooler(embeddings).squeeze(-1)
        return weights



class AllBertWeighter(BaseWeighter):
    def __init__(self, manager):
        super().__init__(manager)

        self.bert = AutoModel.from_pretrained(manager.plm_dir)
        self.bert.pooler = None


    def forward(self, token_id, attn_mask):
        """
        Args:
            token_id: [B, L]
            attn_mask: [B, L]

        Returns:
            weights: [B, L]
        """
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])
        attn_mask = attn_mask.view(-1, original_shape[-1])

        bert_embedding = self.bert(token_id, attention_mask=attn_mask)[0].view(*original_shape, -1)    # B, L, D
        weights = self._compute_weight(bert_embedding)
        return weights



class CnnWeighter(BaseWeighter):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding = nn.Embedding(manager.vocab_size, manager.gate_embedding_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=manager.gate_embedding_dim,
                out_channels=manager.gate_hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.cnn[0].weight)


    def forward(self, token_id, attn_mask):
        """
        Args:
            token_id: [B, L]
            attn_mask: [B, L]

        Returns:
            weights: [B, L]
        """
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])

        token_embedding = self.embedding(token_id)
        cnn_input = token_embedding.transpose(-1, -2)
        conv_embedding = self.cnn(cnn_input).transpose(-1, -2).view(*original_shape, -1)
        weight = self._compute_weight(conv_embedding)
        return weight



class TfmWeighter(BaseWeighter):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding = nn.Embedding(manager.vocab_size, manager.gate_hidden_dim)
        self.tfm = TFMLayer(manager.hidden_dim, manager.head_num, 0.1)


    def forward(self, token_id, attn_mask):
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])
        attn_mask = attn_mask.view(-1, original_shape[-1])

        token_embedding = self.embedding(token_id)
        tfm_embedding = self.tfm(token_embedding, attention_mask=attn_mask).view(*original_shape, -1)
        weights = self._compute_weight(tfm_embedding)
        return weights



class FirstWeighter(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = "First"

    def forward(self, token_id, attn_mask):
        return None