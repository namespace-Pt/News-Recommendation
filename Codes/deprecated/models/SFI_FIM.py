import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SFIModel_pipeline1(nn.Module):
    def __init__(self,hparams,vocab,pipeline=False):
        super().__init__()
        self.name = hparams['name']


        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size =hparams['his_size']
        self.batch_size = hparams['batch_size']
        self.dropout_p = hparams['dropout_p']
        self.pipeline = pipeline
        
        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size']

        self.kernel_size = 3
        self.level = 3
        
        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        self.device = hparams['device']

        # pretrained embedding
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=1,padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=2,padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=3,padding=3)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.filter_num)
        self.DropOut = nn.Dropout(p=self.dropout_p)

        self.query_words = nn.Parameter(torch.randn((1,self.filter_num), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn((1,self.filter_num), requires_grad=True))
        
        self.learningToRank = nn.Linear(self.his_size,1)

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

    def _HDC(self,news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3
        
        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, signal_length, levels(3), filter_num]
        """

        news_embedding_dilations = torch.zeros((news_embedding_set.shape[0],self.signal_length,self.level,self.filter_num),device=self.device)
        
        news_embedding_set = news_embedding_set.transpose(-2,-1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_set)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)        

        news_embedding_d3 = self.CNN_d3(news_embedding_set)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,:,2,:] = self.ReLU(news_embedding_d3)
        
        return news_embedding_dilations
        
    def _news_encoder(self,news_batch):
        """ encode set of news to news representation
        
        Args:
            news_set: tensor of [batch_size, *, signal_length]
        
        Returns:
            news_reprs: tensor of [batch_size, *, self.filter_num]
        """
        news_embedding = self.DropOut(self.embedding[news_batch]).view(-1, self.signal_length, self.embedding_dim)
        news_embedding_dilations = self._HDC(news_embedding).view(self.batch_size, news_batch.shape[1], self.signal_length, self.level, self.filter_num)
        news_embedding_attn = self._scaled_dp_attention(self.query_levels, news_embedding_dilations, news_embedding_dilations).squeeze(dim=-2)
        news_reprs = self._scaled_dp_attention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(dim=-2).view(self.batch_size, news_batch.shape[1], self.filter_num)
        return news_embedding_dilations, news_reprs

    def _click_predictor(self,cdd_news_reprs, his_news_reprs):
        """ calculate batch of click probabolity

        Args:
            cdd_news_reprs: tensor of [batch_size, cdd_size, *]
            his_news_reprs: tensor of [batch_size, his_size, *]

        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        scores = torch.matmul(cdd_news_reprs, his_news_reprs.transpose(-1,-2))
        scores = self.learningToRank(scores).squeeze(dim=-1)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(scores,dim=1)
        else:
            score = torch.sigmoid(scores).squeeze(dim=-1)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        
        if self.pipeline:
            news_embedding, news_repr = self._news_encoder(x['candidate_title'].long().to(self.device))
            return news_embedding, news_repr

        else:
            cdd_news_set = x['candidate_title'].long().to(self.device)
            _, cdd_news_reprs = self._news_encoder(cdd_news_set)
            
            his_news_set = x['clicked_title'].long().to(self.device)
            _, his_news_reprs = self._news_encoder(his_news_set)
        
            score = self._click_predictor(cdd_news_reprs, his_news_reprs)
            return score

class SFIModel_pipeline2(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']


        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size =hparams['his_size']
        self.batch_size = hparams['batch_size']
        
        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size']
        self.filter_num = hparams['filter_num']
        self.k = hparams['k']
        self.level = 3

        self.device = hparams['device']

        news_reprs = torch.load('data/tensors/news_reprs_{}_{}-[{}].tensor'.format(hparams['scale'],hparams['mode'],hparams['name']))
        self.news_repr = nn.Embedding.from_pretrained(news_reprs, freeze=True).to(self.device)
        # self.news_repr = news_reprs.to(self.device)

        news_embeddings = torch.load('data/tensors/news_embeddings_{}_{}-[{}].tensor'.format(hparams['scale'],hparams['mode'],hparams['name']))
        self.news_embedding = nn.Embedding.from_pretrained(news_embeddings.view(news_embeddings.shape[0],-1), freeze=True).to(self.device)
        # self.news_embedding = news_embeddings.view(news_embeddings.shape[0],-1).to(self.device)

        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        self.learningToRank = nn.Linear(self.his_size,1)
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32,kernel_size=[3,3,3],padding=1),
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
        return attn_output.squeeze(dim=-2)

    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_reprs: tensor of [batch_size, cdd_size, signal_length, level, filter_num]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, *], where * is derived from MaxPooling with no padding
        """

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        cdd_news_reprs = cdd_news_reprs.transpose(-2,-3)
        his_news_reprs = his_news_reprs.transpose(-2,-3)

        fusion_tensor = torch.matmul(cdd_news_reprs.unsqueeze(dim=2),his_news_reprs.transpose(-2,-1)) / math.sqrt(self.filter_num)
        # print(fusion_tensor.shape)
        
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
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score).squeeze(dim=-1)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
    
        cdd_news_id = x['cdd_id'].long().to(self.device)
        cdd_repr = self.news_repr(cdd_news_id)#.to(self.device)
        cdd_embedding = self.news_embedding(cdd_news_id).view(self.batch_size, self.cdd_size, self.signal_length, self.level, self.filter_num)
        his_news_id = x['his_id'].long().to(self.device)
        his_repr = self.news_repr(his_news_id)#.to(self.device)
        his_embedding = self.news_embedding(his_news_id).view(self.batch_size, self.his_size, self.signal_length, self.level, self.filter_num)

        attn_weights = self.softmax(torch.bmm(cdd_repr, his_repr.transpose(-1,-2)))
        _, attn_weights_sorted = attn_weights.detach().sort(dim=-1, descending=True)
        attn_focus = F.one_hot(attn_weights_sorted[:,:,:self.k], num_classes=self.his_size).float()

        # [bs, cs, k, sl, 3, fn]
        his_activated = torch.matmul(attn_focus, his_embedding.view(self.batch_size, 1, self.his_size,-1)).view(self.batch_size, self.cdd_size, self.k, self.signal_length, self.level, self.filter_num)

        fusion_tensors = self._fusion(cdd_embedding, his_activated)
        
        score = self._click_predictor(fusion_tensors)
        return score

class SFIModel_gating(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']

        
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        
        self.dropout_p = hparams['dropout_p']
        self.k = hparams['k']

        self.kernel_size = 3
        self.level = 3

        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size']
        self.his_size =hparams['his_size']

        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        self.device = hparams['device']

        # pretrained embedding
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=1,padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.filter_num,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=2,padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.filter_num,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=3,padding=3)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.filter_num)
        self.DropOut = nn.Dropout(p=self.dropout_p)

        self.query_words = nn.Parameter(torch.randn((1,self.filter_num), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn((1,self.filter_num), requires_grad=True))

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
        return attn_output.squeeze(dim=-2)
    
    def _HDC(self,news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3
        
        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros((news_embedding_set.shape[0],self.signal_length,self.level,self.filter_num),device=self.device)
        
        news_embedding_set = news_embedding_set.transpose(-2,-1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_d1.transpose(-2,-1))
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)        

        news_embedding_d3 = self.CNN_d3(news_embedding_d2.transpose(-2,-1))
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,:,2,:] = self.ReLU(news_embedding_d3)
        
        return news_embedding_dilations
        
    def _news_encoder(self,news_batch):
        """ encode set of news to news representation
        
        Args:
            news_set: tensor of [batch_size, *, signal_length]
        
        Returns:
            news_reprs: tensor of [batch_size, *, self.filter_num]
        """
        news_embedding = self.DropOut(self.embedding[news_batch]).view(-1, self.signal_length, self.embedding_dim)
        news_embedding_dilations = self._HDC(news_embedding).view(self.batch_size, news_batch.shape[1], self.signal_length, self.level, self.filter_num)
        news_embedding_attn = self._scaled_dp_attention(self.query_levels, news_embedding_dilations, news_embedding_dilations).squeeze(dim=-2)
        news_reprs = self._scaled_dp_attention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(dim=-2).view(self.batch_size, news_batch.shape[1], self.filter_num)
        return news_embedding_dilations, news_reprs
    
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

        # [bs, cs, k, sl, 3, fn]
        his_activated = torch.matmul(attn_focus, his_embedding.view(self.batch_size, 1, self.his_size,-1)).view(self.batch_size, self.cdd_size, self.k, self.signal_length, self.level, self.filter_num)

        return his_activated

    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_reprs: tensor of [batch_size, cdd_size, signal_length, level, filter_num]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, *], where * is derived from MaxPooling with no padding
        """

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        cdd_news_reprs = cdd_news_reprs.transpose(-2,-3)
        his_news_reprs = his_news_reprs.transpose(-2,-3)

        fusion_tensor = torch.matmul(cdd_news_reprs.unsqueeze(dim=2),his_news_reprs.transpose(-2,-1)) / math.sqrt(self.filter_num)
        # print(fusion_tensor.shape)
        
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
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
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
        
        # compress batch_size and his_size into dim0
        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_reprs = self._news_encoder(his_news)

        # print(cdd_news_embedding.shape, his_news_embedding.shape, cdd_news_reprs.shape, his_news_reprs.shape)

        his_activated = self._news_attention(cdd_news_reprs, his_news_reprs, his_news_embedding, x['his_mask'].to(self.device))
        
        fusion_tensors = self._fusion(cdd_news_embedding, his_activated)
        
        score = self._click_predictor(fusion_tensors)
        return score

class SFIModel_unified(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']

        
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        
        self.dropout_p = hparams['dropout_p']

        self.k = hparams['k']
        self.integration = hparams['integration']

        self.kernel_size = 3
        self.level = 3

        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size']
        self.his_size =hparams['his_size']

        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        self.device = hparams['device']

        # pretrained embedding
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.Softmax(dim=-1)
        
        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=1,padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=2,padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=3,padding=3)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.filter_num)
        self.DropOut = nn.Dropout(p=self.dropout_p)

        self.query_words = nn.Parameter(torch.rand((1,self.filter_num * self.level), requires_grad=True))
        # self.query_words = nn.Parameter(torch.randn((1,self.filter_num), requires_grad=True))

        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3]),
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3])
        )
        
        self.learningToRank_itr = nn.Linear(int((int((self.k - 3)/3 + 1) - 3)/3 + 1) * 2 * 2 * 16,1)
        self.learningToRank_repr = nn.Linear(self.his_size, 1)

        if self.integration == 'gate':
            self.a = nn.Parameter(torch.rand(1), requires_grad=True)

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
    
    def _HDC(self,news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3
        
        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros((news_embedding_set.shape[0],self.signal_length,self.level,self.filter_num),device=self.device)
        
        news_embedding_set = news_embedding_set.transpose(-2,-1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_set)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)        

        news_embedding_d3 = self.CNN_d3(news_embedding_set)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,:,2,:] = self.ReLU(news_embedding_d3)
        
        return news_embedding_dilations
        
    def _news_encoder(self,news_batch):
        """ encode set of news to news representation
        
        Args:
            news_set: tensor of [batch_size, *, signal_length]
        
        Returns:
            news_reprs: tensor of [batch_size, *, self.filter_num]
        """
        news_embedding = self.DropOut(self.embedding[news_batch]).view(-1, self.signal_length, self.embedding_dim)
        news_embedding_dilations = self._HDC(news_embedding).view(self.batch_size, news_batch.shape[1], self.signal_length, self.level, self.filter_num)
        news_embedding = news_embedding_dilations.view(self.batch_size, news_batch.shape[1], self.signal_length, self.level * self.filter_num)
        # news_embedding_attn = self._scaled_dp_attention(self.query_levels, news_embedding_dilations, news_embedding_dilations).squeeze()
        news_reprs = self._scaled_dp_attention(self.query_words, news_embedding, news_embedding).squeeze(dim=-2)
        return news_embedding_dilations, news_reprs
    
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

        # [bs, cs, k, sl, 3, fn]
        his_activated = torch.matmul(attn_focus, his_embedding.view(self.batch_size, 1, self.his_size,-1)).view(self.batch_size, self.cdd_size, self.k, self.signal_length, self.level, self.filter_num)

        return his_activated, attn_weights

    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_reprs: tensor of [batch_size, cdd_size, signal_length, level, filter_num]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, *], where * is derived from MaxPooling with no padding
        """

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        cdd_news_reprs = cdd_news_reprs.transpose(-2,-3)
        his_news_reprs = his_news_reprs.transpose(-2,-3)

        fusion_tensor = torch.matmul(cdd_news_reprs.unsqueeze(dim=2),his_news_reprs.transpose(-2,-1)) / math.sqrt(self.filter_num)
        # print(fusion_tensor.shape)
        
        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, self.k, self.level, self.signal_length, self.signal_length).transpose(1,2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(self.batch_size,self.cdd_size,-1)
        
        return fusion_tensor
    
    def _click_predictor(self, fusion_tensors, attn_weights):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, 320]
        
        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        score_itr = self.learningToRank_itr(fusion_tensors).squeeze(dim=-1)
        score_repr = self.learningToRank_repr(attn_weights).squeeze(dim=-1)
        
        if self.integration == 'harmony':
            score = 2/(1/score_itr + 1/score_repr)
        elif self.integration == 'gate':
            score = self.a * score_itr + (1 - self.a) * score_repr

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
        
        # compress batch_size and his_size into dim0
        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_reprs = self._news_encoder(his_news)

        # print(cdd_news_embedding.shape, his_news_embedding.shape, cdd_news_reprs.shape, his_news_reprs.shape)

        his_activated, attn_weights = self._news_attention(cdd_news_reprs, his_news_reprs, his_news_embedding, x['his_mask'].to(self.device))
        
        fusion_tensors = self._fusion(cdd_news_embedding, his_activated)
        
        score = self._click_predictor(fusion_tensors, attn_weights)
        return score