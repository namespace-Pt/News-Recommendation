import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.Soft_TopK import TopK_custom

class GCAModel_greedy(nn.Module):
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

        self.head_num = hparams['head_num']
        self.query_dim = hparams['query_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.repr_dim = self.head_num * self.value_dim
        
        self.query_words = nn.Parameter(torch.randn((1,self.query_dim), requires_grad=True))

        # Soft-topK module
        # self.topk = TopK_custom(hparams['k'],self.device,epsilon=hparams['epsilon'],max_iter=100)
        self.k = hparams['k']

        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        self.gumbel_softmax = nn.functional.gumbel_softmax
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])
        self.CosSim = nn.CosineSimilarity(dim=-1)

        self.queryProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.value_dim, bias=False) for _ in range(self.head_num)])
        self.keyProject_words = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)

        self.SeqCNN2D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
        )

        self.learningToRank = nn.Linear(144, 1)

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

    def _multi_head_self_attention(self,input,mode):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length/transformer_length, repr_dim]
        
        Returns:
            additive_attn_repr: tensor of [batch_size, *, repr_dim]
            multi_head_self_attn_value: tensor of [batch_size, *, signal_length, repr_dim]

        """
        self_attn_outputs = [self._self_attention(input,i,1) for i in range(self.head_num)]
        multi_head_self_attn_value = torch.cat(self_attn_outputs,dim=-1)
        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))
        additive_attn_repr = self._scaled_dp_attention(self.query_words,multi_head_self_attn_key,multi_head_self_attn_value).squeeze(dim=-2)
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

    def _news_attention(self, cdd_repr, his_repr, his_embedding, his_mask):
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
        # [bs, cs, hs]
        attn_weights = torch.bmm(cdd_repr,his_repr.transpose(-1,-2))

        his_activated_list = []

        # Padding in history will cause 0 in attention weights, underlying the probability that gumbel_softmax may attend to those meaningless 0 vectors. 
        # Masking off these 0s will force the gumbel_softmax to attend to only non-zero histories.
        # Masking in candidate also cause such a problem, however we donot need to fix it
        # because the whole row of attention weight matrix is zero so gumbel_softmax can only capture 0 vectors
        # though reuslting in redundant calculation but it is what we want of padding 0 to negtive sampled candidates as they may be less than npratio.
        attn_weights = attn_weights.masked_fill(his_mask.transpose(-1,-2), -float("inf"))
    
        for i in range(self.k):
            attn_focus = F.gumbel_softmax(attn_weights,dim=-1,tau=0.1,hard=True)
            his_activated = torch.matmul(attn_focus,his_embedding.view(self.batch_size,self.his_size,-1)).view(self.batch_size,self.cdd_size,self.signal_length,self.repr_dim)
            his_activated_list.append(his_activated)
            attn_weights = attn_weights.masked_fill(attn_focus.bool(), -float('inf'))
            
        
        # [bs, cs, k, sl, rd]
        his_activated = torch.stack(his_activated_list, dim=2)
        return his_activated
    
    def _fusion(self,cdd_embedding,his_activated):
        """ fuse activated history news and candidate news into interaction matrix, words in history is column and words in candidate is row, then apply KNRM on each interaction matrix

        Args:
            cdd_embedding: tensor of [batch_size, cdd_size, signal_length, repr_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, repr_dim]
        
        Returns:
            fusion_vectors: tensor of [batch_size, cdd_size, repr_dim]
        """
        # [bs,cs,k,sl,sl]
        fusion_matrices = torch.matmul(cdd_embedding.unsqueeze(dim=2), his_activated.transpose(-2,-1)).view(-1, self.k, self.signal_length, self.signal_length)
        fusion_vectors = self.SeqCNN2D(fusion_matrices).view(self.batch_size, self.cdd_size, -1)
        return fusion_vectors
    
    def _click_predictor(self,fusion_vectors):
        """ calculate batch of click probability              
        Args:
            pooling_vectors: tensor of [batch_size, cdd_size, repr_dim]
        
        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = self.learningToRank(fusion_vectors).squeeze(dim=-1)

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
        # print(cdd_news_repr.shape,cdd_news_embedding_attn.shape)

        # mask the history news 
        his_mask = x['his_mask'].to(self.device)

        his_activated = self._news_attention(cdd_news_repr,his_news_repr,his_news_embedding_attn, his_mask)
        fusion_vectors = self._fusion(cdd_news_embedding_attn,his_activated)

        score_batch = self._click_predictor(fusion_vectors)
        return score_batch