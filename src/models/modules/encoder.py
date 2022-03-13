import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence
from .attention import scaled_dp_attention, extend_attention_mask, TFMLayer



class BaseNewsEncoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = type(self).__name__[:-11]



class BaseUserEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__[:-11]



class CnnNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding_dim = manager.plm_dim
        bert = AutoModel.from_pretrained(manager.plm_dir)
        self.embedding = bert.embeddings.word_embeddings

        self.cnn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=manager.hidden_dim,
            kernel_size=3,
            padding=1
        )
        nn.init.xavier_normal_(self.cnn.weight)

        self.news_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.news_query)
        self.newsProject = nn.Linear(manager.hidden_dim, manager.hidden_dim)
        nn.init.xavier_normal_(self.newsProject.weight)
        self.Tanh = nn.Tanh()
        self.Relu = nn.ReLU()


    def forward(self, token_id, attn_mask, token_weight=None):
        """ encode news through 1-d CNN
        """
        original_shape = token_id.shape
        token_embedding = self.embedding(token_id)
        if token_weight is not None:
            token_embedding = token_embedding * token_weight.unsqueeze(-1)
        cnn_input = token_embedding.view(-1, original_shape[-1], self.embedding_dim).transpose(-2, -1)
        cnn_output = self.Relu(self.cnn(cnn_input)).transpose(-2, -1).view(*original_shape, -1)
        news_embedding = scaled_dp_attention(self.news_query, self.Tanh(self.newsProject(cnn_output)), cnn_output, attn_mask=attn_mask.unsqueeze(-2)).squeeze(dim=-2)
        return cnn_output, news_embedding



class AllBertNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)
        self.plm = AutoModel.from_pretrained(manager.plm_dir)
        self.plm.pooler = None


    def forward(self, token_id, attn_mask):
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])
        attn_mask = attn_mask.view(-1, original_shape[-1])

        token_embedding = self.plm(token_id, attention_mask=attn_mask).last_hidden_state
        news_embedding = token_embedding[:, 0].view(*original_shape[:-1], -1)
        token_embedding = token_embedding.view(*original_shape, -1)
        return token_embedding, news_embedding



class GatedBertNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)
        plm = AutoModel.from_pretrained(manager.plm_dir)
        self.embeddings = plm.embeddings
        self.plm = plm.encoder

        self.news_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.news_query)
        # self.newsProject = nn.Linear(manager.hidden_dim, manager.hidden_dim)
        # nn.init.xavier_normal_(self.newsProject.weight)
        # self.Tanh = nn.Tanh()


    def forward(self, token_id, attn_mask, token_weight=None):
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])
        attn_mask = attn_mask.view(-1, original_shape[-1])

        token_embedding = self.embeddings(token_id)

        if token_weight is not None:
            token_weight = token_weight.view(-1, original_shape[-1]).unsqueeze(-1)
            token_embedding = token_embedding * (token_weight + (1 - token_weight.detach()))

        extended_attn_mask = extend_attention_mask(attn_mask)
        token_embedding = self.plm(token_embedding, attention_mask=extended_attn_mask).last_hidden_state
        # we do not keep [CLS] and [SEP] after gating, so it's better to use attention pooling
        news_embedding = scaled_dp_attention(self.news_query, token_embedding, token_embedding, attn_mask=attn_mask.unsqueeze(-2)).squeeze(dim=-2).view(*original_shape[:-1], -1)
        token_embedding = token_embedding.view(*original_shape, -1)
        return token_embedding, news_embedding



class TfmNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)
        self.embedding_dim = manager.plm_dim
        bert = AutoModel.from_pretrained(manager.plm_dir)
        self.embedding = bert.embeddings.word_embeddings
        self.transformer = TFMLayer(manager.hidden_dim, manager.head_num, 0.1)

        self.news_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.news_query)
        self.newsProject = nn.Linear(manager.hidden_dim, manager.hidden_dim)
        nn.init.xavier_normal_(self.newsProject.weight)
        self.Tanh = nn.Tanh()


    def forward(self, token_id, attn_mask, token_weight=None):
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])
        attn_mask = attn_mask.view(-1, original_shape[-1])

        token_embedding = self.embedding(token_id)
        if token_weight is not None:
            token_weight = token_weight.view(-1, original_shape[-1])
            token_embedding = token_embedding * token_weight.unsqueeze(-1)

        token_embedding = self.transformer(token_embedding, attention_mask=attn_mask)
        news_embedding = scaled_dp_attention(self.news_query, self.Tanh(self.newsProject(token_embedding)), token_embedding, attn_mask=attn_mask.unsqueeze(-2)).squeeze(dim=-2).view(*original_shape[:-1], -1)
        token_embedding = token_embedding.view(*original_shape, -1)
        return token_embedding, news_embedding



class RnnUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()
        # if manager.encoderU == 'gru':
        self.rnn = nn.GRU(manager.hidden_dim, manager.hidden_dim, batch_first=True)
        # elif manager.encoderU == 'lstm':
        #     self.rnn = nn.LSTM(manager.hidden_dim, manager.hidden_dim, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)


    def forward(self, news_embedding, his_mask):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        lens = his_mask.sum(dim=-1).cpu()
        rnn_input = pack_padded_sequence(news_embedding, lens, batch_first=True, enforce_sorted=False)

        _, user_embedding = self.rnn(rnn_input)
        if type(user_embedding) is tuple:
            user_embedding = user_embedding[0]
        return user_embedding.transpose(0,1)



class SumUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()


    def forward(self, news_embedding, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = news_embedding.sum(dim=-2, keepdim=True)
        return user_embedding



class AvgUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()


    def forward(self, news_embedding, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = news_embedding.mean(dim=-2, keepdim=True)
        return user_embedding



class AttnUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()

        self.user_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.user_query)


    def forward(self, news_embedding, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = scaled_dp_attention(self.user_query, news_embedding, news_embedding)
        return user_embedding



class TfmUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()
        self.transformer = TFMLayer(manager.hidden_dim, manager.head_num, 0.1)
        self.user_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.user_query)
        self.userProject = nn.Linear(manager.hidden_dim, manager.hidden_dim)
        nn.init.xavier_normal_(self.userProject.weight)
        self.Tanh = nn.Tanh()


    def forward(self, news_embedding, his_mask):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        news_embedding = self.transformer(news_embedding, attention_mask=his_mask)
        user_embedding = scaled_dp_attention(self.user_query, self.Tanh(self.userProject(news_embedding)), news_embedding, attn_mask=his_mask.unsqueeze(-2))
        return user_embedding

