'''
Author: Pt
Date: 2020-11-14 16:00:00
LastEditTime: 2020-11-20 01:28:10
Description: Implementation of Finegrained Interest Matching method for neural news recommendation 
'''

import torch
import math
import torch.nn as nn

class FIMModel(nn.Module):
    def __init__(self,hparams,vocab,npratio):
        super().__init__()
        self.npratio = npratio
        self.metrics = hparams['metrics']

        self.batch_size = hparams['batch_size']
        self.level = hparams['dilation_level']
        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size'] + 1 + 1
        self.his_size =hparams['his_size']

        self.kernel_size = hparams['kernel_size']
        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # pretrained embedding
        self.embedding = vocab.vectors
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.functional.softmax
        
        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=1,padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.filter_num,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=2,padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.filter_num,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=3,padding=3)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm((self.filter_num,self.signal_length))
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3]),
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3])
        )
        
        self.predictor = nn.Linear(320,1)

    def _HDC(self,news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3
        
        Args:
            news_embedding_set: tensor of [set_size * news_num(npratio + 1/1), signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size * news_num, levels(3), filter_num, signal_length]
        """
        news_embedding_dilations = torch.zeros((news_embedding_set.shape[0],self.level,self.filter_num,self.signal_length),device=self.device)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1)
        news_embedding_dilations[:,0,:,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_d1)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2)
        news_embedding_dilations[:,1,:,:] = self.ReLU(news_embedding_d2)        

        news_embedding_d3 = self.CNN_d3(news_embedding_d2)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3)
        news_embedding_dilations[:,2,:,:] = self.ReLU(news_embedding_d3)
        
        return news_embedding_dilations
        
    def _news_encoder(self,news_set):
        """ encode set of news to news representation of [batch_size * filter_num * signal_length(title_size + category_length + subcategory_length)]
        
        Args:
            news_set: tensor of [set_size, signal_length]
        
        Returns:
            news_embedding_stack: dict of tensor of [set_size, filter_num, signal_length]
        """
        news_embedding = self.embedding[news_set].permute(0,2,1).to(self.device)
        news_embedding_dilations = self._HDC(news_embedding)
        return news_embedding_dilations
    
    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_reprs: tensor of [batch_size * news_num(1), level, filter_num, signal_length]
            his_news_reprs: tensor of [batch_size * his_size, level, filter_num, signal_length]

        Returns:
            fusion_tensor: tensor of [batch_size, 320], where 320 is derived from MaxPooling with no padding
        """

        cdd_news_reprs = torch.repeat_interleave(cdd_news_reprs,repeats=self.his_size,dim=0).view(-1,self.filter_num,self.signal_length)
        fusion_tensor = torch.bmm(his_news_reprs.view(-1,self.filter_num,self.signal_length).permute(0,2,1),cdd_news_reprs) / math.sqrt(self.filter_num)

        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(self.batch_size,self.his_size,self.level,self.signal_length,self.signal_length).permute(0,2,1,3,4)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(self.batch_size,-1)
        return fusion_tensor
    
    def _click_predictor(self,fusion_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, npratio + 1, 320]
        
        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        score = self.predictor(fusion_tensors)
        if self.npratio > 0:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self,x):
        cdd_news_set = torch.cat([x['candidate_title'].long().to(self.device),x['candidate_category'].long().to(self.device),x['candidate_subcategory'].long().to(self.device)],dim=2).view(-1,self.signal_length)

        cdd_news_reprs = self._news_encoder(cdd_news_set).view(self.batch_size,-1,self.level,self.filter_num,self.signal_length)

        # compress batch_size and his_size into dim0
        his_news_set = torch.cat([x['clicked_title'].long().to(self.device),x['clicked_category'].long().to(self.device),x['clicked_subcategory'].long().to(self.device)],dim=2).view(-1,self.signal_length)

        his_news_reprs = self._news_encoder(his_news_set)
        
        if self.npratio > 0:
            # 320 is derived from maxpooling in SeqCNN3D
            fusion_tensors = torch.zeros((self.batch_size, self.npratio + 1, 320),device=self.device)

            for cdd_idx in range(self.npratio + 1):
                fusion_tensors[:,cdd_idx,:] = self._fusion(cdd_news_reprs[:,cdd_idx,:,:,:],his_news_reprs)
        
        else:
            fusion_tensors = self._fusion(cdd_news_reprs[:,0,:,:,:],his_news_reprs)
            

        score = self._click_predictor(fusion_tensors).squeeze()
        
        return score