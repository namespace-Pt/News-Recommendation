import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SFIModel_gating(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        
        self.dropout_p = hparams['dropout_p']

        self.k = hparams['k']
        
        self.level = hparams['head_num']

        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']

        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']

        self.device = hparams['device']

        # pretrained embedding
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.DropOut = nn.Dropout(p=self.dropout_p)

        self.query_words = nn.Parameter(torch.randn((1, self.query_dim), requires_grad=True))
        # self.query_words = nn.Parameter(torch.randn((1,self.filter_num), requires_grad=True))

        self.queryProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.embedding_dim, bias=False) for _ in range(self.level)])
        self.valueProject_words = nn.ModuleList([]).extend([nn.Linear(self.embedding_dim,self.value_dim, bias=False) for _ in range(self.level)])

        self.keyProject_words = nn.Linear(self.value_dim * self.level, self.query_dim, bias=True)

        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=self.level,out_channels=32,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3]),
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3])
        )
        
        self.learningToRank = nn.Linear(int((int((self.k - 3)/3 + 1) - 3)/3 + 1) * 2 * 2 * 16,1)

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
        return attn_output

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

    def _multi_head_self_attention(self,input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, repr_dim]
        
        Returns:
            additive_attn_repr: tensor of [batch_size, *, repr_dim]
            multi_head_self_attn_value: tensor of [batch_size, *, signal_length, repr_dim]

        """
        self_attn_outputs = [self._self_attention(input,i) for i in range(self.level)]
        mha_embedding = torch.stack(self_attn_outputs, dim=-3)
        mha_repr = torch.cat(self_attn_outputs, dim=-1)

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_key = torch.tanh(self.keyProject_words(mha_repr))
        additive_attn_repr = self._scaled_dp_attention(self.query_words,multi_head_self_attn_key,mha_repr).squeeze(dim=-2)
        return mha_embedding, additive_attn_repr
    
    def _news_encoder(self,news_batch):
        """ encode set of news to news representation
        
        Args:
            news_set: tensor of [batch_size, *, signal_length]
        
        Returns:
            news_embedding_dilations: tensor of [batch_size, *, level, signal_length, filter_num]
        """
        news_embedding = self.DropOut(self.embedding[news_batch])
        news_embedding_attn, news_reprs = self._multi_head_self_attention(news_embedding)

        return news_embedding_attn, news_reprs
    
    def _news_attention(self, cdd_repr, his_repr, his_embedding, his_mask):
        """ apply news-level attention

        Args:
            cdd_repr: tensor of [batch_size, cdd_size, *]
            his_repr: tensor of [batch_size, his_size, *]
            his_embedding: tensor of [batch_size, his_size, self.level, signal_length, *]
            his_mask: tensor of [batch_size, his_size, 1]

        Returns:
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, *]
        """
        # [bs, cs, hs]
        attn_weights = torch.bmm(cdd_repr,his_repr.transpose(-1,-2))

        # his_activated_list = []

        # Padding in history will cause 0 in attention weights, underlying the probability that gumbel_softmax may attend to those meaningless 0 vectors. 
        # Masking off these 0s will force the gumbel_softmax to attend to only non-zero histories.
        # Masking in candidate also cause such a problem, however we donot need to fix it
        # because the whole row of attention weight matrix is zero so gumbel_softmax can only capture 0 vectors
        # though reuslting in redundant calculation but it is what we want of padding 0 to negtive sampled candidates as they may be less than npratio.
        attn_weights = self.softmax(attn_weights.masked_fill(his_mask.transpose(-1,-2), -float("inf")))
        # print(attn_weights.shape, attn_weights[0,0])
        
        # for i in range(self.k):
        #     # attn_focus = F.gumbel_softmax(attn_weights,dim=-1,tau=0.1,hard=True)
        #     attn_focus = F.one_hot(attn_weights.argmax(dim=-1), num_classes=self.his_size).float()
        #     his_activated = torch.matmul(attn_focus,his_embedding.view(self.batch_size,self.his_size,-1)).view(self.batch_size, self.cdd_size, self.level, self.signal_length, self.filter_num)
        #     his_activated_list.append(his_activated)
        #     attn_weights = attn_weights.masked_fill(attn_focus.bool(), -float('inf'))

        _, attn_weights_sorted = attn_weights.detach().sort(dim=-1, descending=True)
        attn_focus = F.one_hot(attn_weights_sorted[:,:,:self.k], num_classes=self.his_size).float()

        # [bs, cs, k, sl, rd]
        his_activated = torch.matmul(attn_focus, his_embedding.view(self.batch_size, 1, self.his_size,-1)).view(self.batch_size, self.cdd_size, self.k, self.level, self.signal_length, self.value_dim)
        
        # [bs, cs, k, sl, rd]
        # his_activated = torch.stack(his_activated_list, dim=2)
        return his_activated

    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, level, signal_length, filter_num]
            his_activated: tensor of [batch_size, cdd_size, k, level, signal_length, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, 320], where 320 is derived from MaxPooling with no padding
        """

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_reprs.unsqueeze(dim=2),his_news_reprs.transpose(-2,-1)) / math.sqrt(self.value_dim)
        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, self.k, self.level, self.signal_length, self.signal_length).transpose(1,2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(self.batch_size,self.cdd_size,-1)
        
        return fusion_tensor
    
    def _click_predictor(self,fusion_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, 320]
        
        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        score = self.learningToRank(fusion_tensors)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        # compress batch_size and cdd_size into dim0
        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, cdd_news_reprs = self._news_encoder(cdd_news)
        # print(cdd_news_embedding.shape, cdd_news_reprs.shape)
        
        # compress batch_size and his_size into dim0
        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_reprs = self._news_encoder(his_news)
        # print(his_news_embedding.shape, his_news_reprs.shape)

        his_activated = self._news_attention(cdd_news_reprs, his_news_reprs, his_news_embedding, x['his_mask'].to(self.device))
        
        fusion_tensors = self._fusion(cdd_news_embedding, his_activated)
        
        score = self._click_predictor(fusion_tensors).squeeze()
        return score