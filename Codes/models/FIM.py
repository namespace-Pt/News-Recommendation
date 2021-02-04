import torch
import math
import torch.nn as nn

class FIMModel(nn.Module):
    def __init__(self,hparams,vocab):
        super().__init__()
        self.name = hparams['name']
        self.metrics = hparams['metrics']

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size =hparams['his_size']
        self.batch_size = hparams['batch_size']
        self.dropout_p = hparams['dropout_p']
        
        # concatenate category embedding and subcategory embedding
        self.signal_length = hparams['title_size']# + 1 + 1

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
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3]),
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3])
        )
        
        self.learningToRank = nn.Linear(int(int(self.signal_length/3)/3) ** 2 * int(int(self.his_size/3)/3) * 16,1)

    def _HDC(self,news_embedding):
        """ stack 1d CNN with dilation rate expanding from 1 to 3
        
        Args:
            news_embedding: tensor of [batch_size, *, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [batch_size, *, levels(3), signal_length, filter_num]
            news_reprs: tensor  of [bath_size, *, filter_num (* levels)]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros((self.batch_size * news_embedding.shape[1], self.level, self.signal_length, self.filter_num),device=self.device)
        
        news_embedding_set = news_embedding.transpose(-2,-1).view(-1, self.embedding_dim, self.signal_length)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,0,:,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_d2(news_embedding_set)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,1,:,:] = self.ReLU(news_embedding_d2)        

        news_embedding_d3 = self.CNN_d3(news_embedding_set)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,2,:,:] = self.ReLU(news_embedding_d3)

        return news_embedding_dilations.view(self.batch_size, news_embedding.shape[1], self.level, self.signal_length, self.filter_num)
        
    def _news_encoder(self,news_batch):
        """ encode set of news to news representation
        
        Args:
            news_set: tensor of [batch_size, *, signal_length]
        
        Returns:
            news_embedding_dilations: tensor of [batch_size, *, level, signal_length, filter_num]
        """
        news_embedding = self.DropOut(self.embedding[news_batch])
        news_embedding_dilations = self._HDC(news_embedding)

        return news_embedding_dilations
    
    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_reprs: tensor of [batch_size, cdd_size, level, signal_length, filter_num]
            his_news_reprs: tensor of [batch_size, his_size, level, signal_length, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, 320], where 320 is derived from MaxPooling with no padding
        """

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_reprs.unsqueeze(dim=2),his_news_reprs.unsqueeze(dim=1).transpose(-2,-1)) / math.sqrt(self.filter_num)
        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, self.his_size, self.level, self.signal_length, self.signal_length).transpose(1,2)

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
        # compress batch_size and cdd_size into dim0
        cdd_news_set = x['candidate_title'].long().to(self.device)
        cdd_news_reprs = self._news_encoder(cdd_news_set)
        
        # compress batch_size and his_size into dim0
        his_news_set = x['clicked_title'].long().to(self.device)
        his_news_reprs = self._news_encoder(his_news_set)
                
        fusion_tensors = self._fusion(cdd_news_reprs, his_news_reprs)
            
        score = self._click_predictor(fusion_tensors).squeeze()
        return score