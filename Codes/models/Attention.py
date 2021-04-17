import math
import torch
import torch.nn as nn

class Attention():
    @staticmethod
    def ScaledDpAttention(query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        attn_weights = nn.functional.softmax(attn_weights,dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output