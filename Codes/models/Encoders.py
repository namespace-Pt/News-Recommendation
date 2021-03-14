import math
import torch
import torch.nn as nn


class MHA_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'mha-encoder'

        self.level = 1

        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']
        self.head_num = hparams['head_num']

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.device = hparams['device']

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors).to(self.device)

        self.softmax = nn.Softmax(dim=-1)
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query = nn.Parameter(torch.randn(
            (1, self.query_dim), requires_grad=True))
        self.queryProjects = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.head_num)])
        self.valueProjects = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.value_dim) for _ in range(self.head_num)])
        self.keyProject = nn.Linear(self.hidden_dim, self.query_dim)
        self.valueProject = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [*, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _self_attention(self, input, head_idx):
        """ apply self attention of head#idx over input tensor

        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        query = self.queryProjects[head_idx](input)
        attn_output = self._scaled_dp_attention(query, input, input)
        self_attn_output = self.valueProjects[head_idx](attn_output)
        return self_attn_output

    def _multi_head_self_attention(self, input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, repr_dim]

        Returns:
            additive_attn_repr: tensor of [batch_size, *, repr_dim]
            multi_head_self_attn_value: tensor of [batch_size, *, signal_length, repr_dim]

        """
        self_attn_outputs = [self._self_attention(
            input, i) for i in range(self.head_num)]
        mha_embedding_native = torch.cat(self_attn_outputs, dim=-1)
        mha_embedding = self.valueProject(mha_embedding_native)

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_key = torch.tanh(
            self.keyProject(mha_embedding_native))
        mha_repr = self._scaled_dp_attention(
            self.query, multi_head_self_attn_key, mha_embedding_native).squeeze(dim=-2)
        return mha_embedding, mha_repr

    def forward(self, news_batch, user_index):
        """ encode news

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding_pretrained = self.DropOut(self.embedding(news_batch))
        news_embedding, news_repr = self._multi_head_self_attention(
            news_embedding_pretrained)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr


class FIM_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'fim-encoder'

        self.kernel_size = 3
        self.level = 3

        # concatenate category embedding and subcategory embedding
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        self.device = hparams['device']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors).to(self.device)

        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=self.level, out_channels=32,
                      kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16,
                      kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [*, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _HDC(self, news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3

        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros(
            (news_embedding_set.shape[0], news_embedding_set.shape[1], self.level, self.hidden_dim), device=self.device)

        news_embedding_set = news_embedding_set.transpose(-2, -1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2, -1))
        news_embedding_dilations[:, :, 0, :] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_d1.transpose(-2, -1))
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2, -1))
        news_embedding_dilations[:, :, 1, :] = self.ReLU(news_embedding_d2)

        news_embedding_d3 = self.CNN_d3(news_embedding_d2.transpose(-2, -1))
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2, -1))
        news_embedding_dilations[:, :, 2, :] = self.ReLU(news_embedding_d3)

        return news_embedding_dilations

    def forward(self, news_batch, user_index):
        """ encode set of news to news representation

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding = self.DropOut(
            self.embedding(news_batch)).view(-1, news_batch.shape[2], self.embedding_dim)
        news_embedding_dilations = self._HDC(news_embedding).view(
            news_batch.shape + (self.level, self.hidden_dim))
        news_embedding_attn = self._scaled_dp_attention(
            self.query_levels, news_embedding_dilations, news_embedding_dilations).squeeze(dim=-2)
        news_reprs = self._scaled_dp_attention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(
            dim=-2).view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)
        return news_embedding_dilations, news_reprs

class NPA_Encoder(nn.Module):
    def __init__(self,hparams,vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'
        
        self.dropout_p = hparams['dropout_p']

        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['query_dim']

        self.device = torch.device(hparams['device'])

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors).to(self.device)
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1,self.user_dim)
        # project e_u to word query preference vector of query_dim
        self.wordQueryProject = nn.Linear(self.user_dim,self.query_dim)
        # project preference query to vector of hidden_dim
        self.wordPrefProject = nn.Linear(self.query_dim,self.hidden_dim)


        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.hidden_dim,kernel_size=3,padding=1)
        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [*, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output
    
    def forward(self, news_batch, user_index):
        """ encode news through 1-d CNN and combine embeddings with personalized attention
        
        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(user_index))
        word_query = self.Tanh(self.wordPrefProject(self.RELU(self.wordQueryProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2,-1)
        news_embedding = self.RELU(self.CNN(news_embedding_pretrained)).transpose(-2,-1).view(news_batch.shape + (self.hidden_dim,))

        news_repr = self._scaled_dp_attention(word_query.view(word_query.shape[0],1,1,word_query.shape[-1]), news_embedding, news_embedding).squeeze(dim=-2)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr