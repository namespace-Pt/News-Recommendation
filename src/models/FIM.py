import torch.nn as nn
from .BaseModel import OneTowerBaseModel
from .modules.encoder import HDCNNNewsEncoder



class FIM(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.encoder = HDCNNNewsEncoder(manager)

        self.seqConv3D = nn.Sequential(
            nn.Conv3d(in_channels=self.encoder.level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )
        nn.init.xavier_normal_(self.seqConv3D[0].weight)
        nn.init.xavier_normal_(self.seqConv3D[3].weight)


        final_dim = (self.his_size // 3 // 3) * (self.sequence_length // 3 // 3) ** 2 * 16
        self.pooler = nn.Linear(final_dim, 1)
        nn.init.xavier_normal_(self.pooler.weight)


    def infer(self, x):
        cdd_token_id = x["cdd_token_id"].to(self.device)
        cdd_token_embedding, _ = self.encoder(cdd_token_id) # B, C, V, L, D

        his_token_id = x["his_token_id"].to(self.device)
        his_token_embedding, _ = self.encoder(his_token_id) # B, N, V, L, D

        cdd_token_embedding = cdd_token_embedding.unsqueeze(2)
        his_token_embedding = his_token_embedding.unsqueeze(1)

        matching = cdd_token_embedding.matmul(his_token_embedding.transpose(-1, -2))    # B, C, N, V, L, L
        B, C, N, V, L = matching.shape[:-1]
        cnn_input = matching.view(-1, N, V, L, L).transpose(1, 2)    # B*C, V, N, L, L
        cnn_output = self.seqConv3D(cnn_input).view(B, C, -1)  # B*C, x

        logits = self.pooler(cnn_output).squeeze(-1)
        return logits


    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss