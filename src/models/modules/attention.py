import math
import torch
from torch import _softmax_backward_data, nn



def scaled_dp_attention(query, key, value, attn_mask=None, return_prob=False):
    """ calculate scaled attended output of values
    Args:
        query: tensor of []
        key: tensor of [batch_size, *, key_num, key_dim]
        value: tensor of [batch_size, *, key_num, value_dim]
        attn_mask: tensor of [batch_size, *, query_num, key_num]
    Returns:
        attn_output: tensor of [batch_size, *, query_num, value_dim]
    """

    # make sure dimension matches
    assert query.shape[-1] == key.shape[-1]
    key = key.transpose(-2, -1)

    attn_score = torch.matmul(query, key)/math.sqrt(query.shape[-1])

    if attn_mask is not None:
        attn_prob = XSoftmax.apply(attn_score, attn_mask, -1)
    else:
        attn_prob = torch.softmax(attn_score, -1)

    attn_output = torch.matmul(attn_prob, value)

    if return_prob:
        return attn_output, attn_prob
    else:
        return attn_output


def extend_attention_mask(attention_mask, reverse=True):
    """
    Args:
        attention_mask (`torch.Tensor`): An attention mask.
    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    else:
        raise NotImplementedError

    if reverse:
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e5

    return extended_attention_mask



class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory
    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None



class TFMSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_p):
        super().__init__()

        self.num_attention_heads = head_num
        self.attention_head_size = int(hidden_dim / head_num)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        if self.all_head_size != hidden_dim:
            RuntimeWarning(f"Truncating given hidden dim {hidden_dim} to {self.all_head_size} so that it can be divided by head num {head_num}")

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout_p)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # broadcast attention masks
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = torch.softmax(attention_scores + attention_mask, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TFMSelfOutput(nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFMAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_p):
        super().__init__()
        self.self = TFMSelfAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_p=dropout_p)
        self.output = TFMSelfOutput(hidden_dim=hidden_dim, dropout_p=dropout_p)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask
        )
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class TFMIntermediate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.intermediate_act_fn = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TFMOutput(nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.dense = nn.Linear(4 * hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFMLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_p):
        """
        hidden_dim: transformer model dimension
        head_num: number of self attention heads
        dropout_p: dropout probability
        """
        super().__init__()
        self.attention = TFMAttention(hidden_dim=hidden_dim, head_num=head_num, dropout_p=dropout_p)
        self.intermediate = TFMIntermediate(hidden_dim=hidden_dim)
        self.output = TFMOutput(hidden_dim=hidden_dim, dropout_p=dropout_p)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        attention_mask = extend_attention_mask(attention_mask)
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
