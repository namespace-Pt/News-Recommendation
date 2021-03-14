import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.Soft_TopK import TopK_custom

class GCAModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1

        self.device = torch.device(hparams['device'])
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']

        mus = torch.arange(-0.9,1.1,0.1,device=self.device)
        self.kernel_num = len(mus)
        self.mus = mus.view(1,1,1,1,1,-1)
        self.sigmas = torch.tensor([0.1]*(self.kernel_num - 1) + [0.001], device=self.device).view(1,1,1,1,1,-1)

        self.head_num = hparams['head_num']
        self.query_dim = hparams['query_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.repr_dim = self.head_num * self.value_dim
        
        self.query_words = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))

        # Soft-topK module
        # self.topk = TopK_custom(hparams['k'],self.device,epsilon=hparams['epsilon'],max_iter=100)

        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])
        self.CosSim = nn.CosineSimilarity(dim=-1)

        self.queryProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.keyProject_words = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.learningToRank = nn.Linear(self.kernel_num, 1)

    def _scaled_dp_attention(self,query,key,value):
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
        key = key.transpose(-2,-1)

        attn_weights = torch.matmul(query,key)/torch.sqrt(torch.tensor([self.embedding_dim],dtype=torch.float,device=self.device))
        attn_weights = self.softmax(attn_weights)
        
        attn_output = torch.matmul(attn_weights,value)

        return attn_output.squeeze(dim=-2)

    def _self_attention(self,input,head_idx):
        """ apply self attention of head#idx over input tensor
        
        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        query = self.queryProject_words[head_idx](input)

        attn_output = self._scaled_dp_attention(query,input,input)
        self_attn_output = self.valueProject_words[head_idx](attn_output)

        return self_attn_output
    
    def _word_attention(self, query, key, value):
        """ apply word-level attention

        Args:
            query: tensor of [1, query_dim]
            key: tensor of [batch_size, *, signal_length, query_dim]
            value: tensor of [batch_size, *, signal_length, repr_dim]

        Returns:
            attn_output: tensor of [batch_size, *, repr_dim]
        """

        attn_output = self._scaled_dp_attention(query,key,value).squeeze(dim=-2)

        return attn_output.squeeze(dim=-2)


    def _multi_head_self_attention(self,input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
        
        Returns:
            multi_head_self_attn: tensor of [batch_size, *, 1, repr_dim]
        """
        self_attn_outputs = [self._self_attention(input,i) for i in range(self.head_num)]

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
        multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))

        additive_attn_repr = self._word_attention(self.query_words, multi_head_self_attn_key,multi_head_self_attn_value)
        return additive_attn_repr, multi_head_self_attn_value

    def _news_encoder(self,news_batch):
        """ encode set of news to news representations of [batch_size, cdd_size, tranformer_dim]
        
        Args:
            news_batch: tensor of [batch_size, cdd_size, title_size]
            word_query: tensor of [set_size, preference_dim]
        
        Returns:
            news_repr: tensor of [batch_size, cdd_size, repr_dim]
            news_embedding_attn: tensor of [batch_size, cdd_size, signal_length, repr_dim] 
        """
        news_embedding = self.DropOut(self.embedding[news_batch])
        news_repr, news_embedding_attn = self._multi_head_self_attention(news_embedding)
        return news_repr, news_embedding_attn

    def _news_attention(self, cdd_repr, his_repr, his_embedding, his_mask, his_title_pad):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, repr_dim]
            his_repr: tensor of [batch_size, his_size, repr_dim]
            his_embedding: tensor of [batch_size, his_size, signal_length, repr_dim]
            his_mask: tensor of [batch_size, his_size, 1]
            his_title_pad: tensor of [batch_size, his_size, signal_length]

        Returns:
            his_activated: tensor of [batch_size, cdd_size, signal_length, embedding_dim]
            his_title_pad: tensor of [batch_size, cdd_size, signal_length]
        """
        attn_weights = torch.bmm(cdd_repr,his_repr.transpose(-1,-2))

        # Padding in history will cause 0 in attention weights, underlying the probability that gumbel_softmax may attend to those meaningless 0 vectors. 
        # Masking off these 0s will force the gumbel_softmax to attend to only non-zero histories.
        # Masking in candidate also cause such a problem, however we donot need to fix it
        # because the whole row of attention weight matrix is zero so gumbel_softmax can only capture 0 vectors
        # though reuslting in redundant calculation but it is what we want of padding 0 to negtive sampled candidates as they may be less than npratio.
        attn_weights = attn_weights.masked_fill(his_mask.transpose(-1,-2), -float("inf"))

        # [batch_size, cdd_size, his_size]
        # attn_focus = self.topk(-attn_weights.view(-1,self.his_size)).view(self.batch_size,self.cdd_size,self.his_size)
        attn_focus = self.gumbel_softmax(attn_weights,dim=-1,tau=0.1,hard=True)

        "5 iteration maybe better?"

        # [batch_size * cdd_size, signal_length * embedding_dim]
        his_activated = torch.matmul(attn_focus,his_embedding.view(self.batch_size,self.his_size,-1)).view(self.batch_size,self.cdd_size,self.signal_length,self.repr_dim)

        # his_pad = torch.matmul(attn_focus, his_title_pad)

        return his_activated#his_activated, his_pad
    
    def _fusion(self,his_activated,cdd_org):
        """ fuse activated history news and candidate news into interaction matrix, words in history is column and words in candidate is row, then apply KNRM on each interaction matrix

        Args:
            his_activated: tensor of [batch_size, cdd_size, signal_length, embedding_dim]
            cdd_org: tensor of [batch_size, cdd_size, signal_length, embedding_dim]
        
        Returns:
            fusion_matrices: tensor of [batch_size, cdd_size, signal_length, signal_length]
        """

        his_activated = F.normalize(his_activated, dim=-1)
        cdd_org = F.normalize(cdd_org, dim=-1)
        fusion_matrices = torch.matmul(cdd_org, his_activated.transpose(-1,-2))
        return fusion_matrices
    
    def _kernel_pooling(self, matrices, mask_cdd, mask_his):
        """
            apply kernel pooling on matrix, in order to get the relatedness from many levels
        
        Args:
            matrices: tensor of [batch_size, cdd_size, signal_length, signal_length, 1]
            mask_cdd: tensor of [batch_size, cdd_size, signal_length, 1]
            mask_his: tensor of [batch_size, cdd_size, 1, signal_length, 1]
        
        Returns:
            pooling_vectors: tensor of [batch_size, cdd_size, kernel_num]
        """
        # tensor method
        pooling_matrices = torch.exp(-(matrices - self.mus) ** 2 / (2 * self.sigmas ** 2)) * mask_his
        pooling_sum = torch.sum(pooling_matrices, dim=3)
        
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * mask_cdd * 0.01
        pooling_vectors = torch.sum(log_pooling_sum, dim=-2)
        return pooling_vectors
    
    def _click_predictor(self,pooling_vectors):
        """ calculate batch of click probability              
        Args:
            pooling_vectors: tensor of [batch_size, cdd_size, kernel_num]
        
        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = self.learningToRank(pooling_vectors).squeeze(dim=-1)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        cdd_news_repr,cdd_news_embedding_attn = self._news_encoder(x['candidate_title'].long().to(self.device))
        his_news_repr,his_news_embedding_attn = self._news_encoder(x['clicked_title'].long().to(self.device))

        # mask the history news 
        his_mask = x['his_mask'].to(self.device)
        cdd_title_pad = x['candidate_title_pad'].float().to(self.device)
        his_title_pad = x['clicked_title_pad'].float().to(self.device)

        attn_weights = self._news_attention(cdd_news_repr,his_news_repr,his_news_embedding_attn, his_mask, his_title_pad)
        # fusion_matrices = self._fusion(his_activated,cdd_news_embedding_origin)
        
        # pooling_vectors = self._kernel_pooling(fusion_matrices.unsqueeze(dim=-1), cdd_title_pad.view(self.batch_size, self.cdd_size, self.signal_length, 1), his_pad.view(self.batch_size, self.cdd_size, 1, self.signal_length, 1))
        # # print(pooling_vectors)
        # score_batch = self._click_predictor(pooling_vectors)
        return attn_weights