import math
import torch
import torch.nn as nn

from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_func
    from flash_attn.bert_padding import unpad_input, pad_input
except:
    pass


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, key_padding_mask=None, causal=False, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda

        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        if key_padding_mask is None:
            q = rearrange(q, 'b s ... -> (b s) ...')
            k = rearrange(k, 'b s ... -> (b s) ...')
            v = rearrange(v, 'b s ... -> (b s) ...')
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
            cu_seqlens_kv = torch.arange(0, (batch_size + 1) * seqlen_kv, step=seqlen_kv, dtype=torch.int32, device=k.device)
            output = flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, seqlen_q, seqlen_kv, self.dropout_p if self.training else 0.0, 
                                              softmax_scale=self.softmax_scale, causal=causal)
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = q.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

            k = rearrange(k, 'b s h d -> b s (h d)')
            v = rearrange(v, 'b s h d -> b s (h d)')
            k_unpad, indices, cu_seqlens_kv, seqlen_kv = unpad_input(k, key_padding_mask)
            v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
            k_unpad = rearrange(k_unpad, 'nnz (h d) -> nnz h d', h=nheads)
            v_unpad = rearrange(v_unpad, 'nnz (h d) -> nnz h d', h=nheads)
            output = flash_attn_unpadded_func(q, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_kv, seqlen_q, seqlen_kv, self.dropout_p if self.training else 0.0, 
                                              softmax_scale=self.softmax_scale, causal=causal)
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)
        query = rearrange(query, "b s (h d) -> b s h d", h=self.num_heads)
        key = rearrange(key, "b s (h d) -> b s h d", h=self.num_heads)
        value = rearrange(value, "b s (h d) -> b s h d", h=self.num_heads)
        # The key_paddding_mask of flash attention is inverse of nn.MultiheadAttention
        if key_padding_mask is not None:
            key_padding_mask = torch.logical_not(key_padding_mask)
        context, attn_weights = self.inner_attn(query, key, value, key_padding_mask=key_padding_mask, causal=self.causal, need_weights=need_weights)
        if need_weights:
            return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
        else:
            return self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))
