import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def kernel_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma


class FIM_Interactor(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.name = 'fim'

        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=level, out_channels=32, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )
        nn.init.xavier_normal_(self.SeqCNN3D[0].weight)
        nn.init.xavier_normal_(self.SeqCNN3D[3].weight)


    def forward(self, cdd_news_embedding, his_activated, **kwargs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, *], where * is derived from MaxPooling with no padding
        """
        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        # [bs, cs, k, lv, sl, hd]
        his_news_embedding = his_activated.transpose(-2, -3)

        # [batch_size, cdd_size, k, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, his_news_embedding.shape[2], his_news_embedding.shape[3],
                                           his_news_embedding.shape[-2], his_news_embedding.shape[-2]).transpose(1, 2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(cdd_news_embedding.shape[0], cdd_news_embedding.shape[1], -1)
        return fusion_tensor


class KNRM_Interactor(nn.Module):
    def __init__(self, kernel_num=11):
        super().__init__()
        self.name = 'knrm'

        mus = torch.tensor(kernel_mus(kernel_num), dtype=torch.float)
        self.hidden_dim = len(mus)
        self.mus = nn.Parameter(mus.view(1,1,1,1,1,1,-1), requires_grad=False)
        self.sigmas = nn.Parameter(torch.tensor(kernel_sigmas(kernel_num), dtype=torch.float).view(1,1,1,1,1,1,-1), requires_grad=False)

    def forward(self, cdd_news_embedding, his_activated, **kwargs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, hidden_dim]
            his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]

        Returns:
            fusion_tensor: tensor of [batch_size, cdd_size, *], where * is derived from MaxPooling with no padding
        """
        cdd_news_embedding = F.normalize(cdd_news_embedding.transpose(-2, -3), dim=-1)
        his_news_embedding = F.normalize(his_activated.transpose(-2, -3), dim=-1)

        # print(cdd_news_embedding.shape, type(cdd_news_embedding), his_news_embedding.shape)

        # [bs, cs, k, lv, sl, sl, 1]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(
            dim=2), his_news_embedding.transpose(-2, -1)) / math.sqrt(cdd_news_embedding.shape[-1])

        # print((torch.abs(fusion_tensor) > 1).any(), fusion_tensor)

        fusion_tensor = fusion_tensor.unsqueeze(dim=-1)

        pooling_matrices = torch.exp(-((fusion_tensor - self.mus) ** 2) / (2 * self.sigmas ** 2)) * kwargs['his_pad']
        pooling_sum = torch.sum(pooling_matrices, dim=-2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * kwargs['cdd_pad'] * 0.01
        pooling_vectors = torch.sum(log_pooling_sum, dim=-2)

        fusion_tensor = torch.sum(torch.sum(pooling_vectors, dim=-2), dim=-2)

        return fusion_tensor