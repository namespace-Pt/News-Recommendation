import torch
import math
import torch.nn as nn

class FIMModel(nn.Module):
    def __init__(self,hparams,encoder):
        super().__init__()

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size =hparams['his_size']
        self.batch_size = hparams['batch_size']

        self.signal_length = hparams['title_size']

        self.kernel_size = 3

        self.encoder = encoder
        self.hidden_dim = self.encoder.hidden_dim
        self.level = self.encoder.level
        self.DropOut = self.encoder.DropOut
        self.name = 'fim-' + encoder.name


        self.device = hparams['device']

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3]),
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3])
        )

        self.learningToRank = nn.Linear(int((int((self.his_size - 3)/3 + 1) - 3)/3 + 1) * 2 * 2 * 16,1)

    def _fusion(self,cdd_news_embedding,his_news_embedding):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, filter_num]
            his_news_embedding: tensor of [batch_size, his_size, signal_length, level, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, 320], where 320 is derived from MaxPooling with no padding
        """

        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        his_news_embedding = his_news_embedding.transpose(-2, -3)

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(dim=2),his_news_embedding.unsqueeze(dim=1).transpose(-2,-1)) / math.sqrt(self.hidden_dim)
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
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, cdd_news_repr = self.encoder(
            cdd_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['cdd_id'].long().to(self.device))
            # attn_mask=x['candidate_title_pad'].to(self.device))

        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, his_news_repr = self.encoder(
            his_news,
            user_index=x['user_index'].long().to(self.device),
            news_id=x['his_id'].long().to(self.device))
            # attn_mask=x['clicked_title_pad'].to(self.device))

        fusion_tensors = self._fusion(cdd_news_embedding, his_news_embedding)

        score = self._click_predictor(fusion_tensors)
        return score