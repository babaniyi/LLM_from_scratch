import numpy as np
from typing import Union

np.random.seed(42)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
  

class SelfAttention():

    def __init__(self, d_in, d_out):

        self.d_out = d_out
        self.W_query = np.random.rand(d_in, d_out)
        self.W_key   = np.random.rand(d_in, d_out)
        self.W_value = np.random.rand(d_in, d_out)

    def forward(self, x):
        # Calculate keys, queries, and values
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # Calculate attention scores and apply softmax
        attn_scores = queries @ keys.T
        #attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True)) / np.sum(np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True)), axis=-1, keepdims=True)

        #attn_weights = softmax(attn_scores / np.sqrt(self.d_out))
        attn_weights = softmax(attn_scores / np.sqrt(keys.shape[-1]))

        # Calculate context vector
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention_np:

    def __init__(self, d_in, d_out, context_length, dropout):
        self.d_out = d_out
        self.W_query = np.random.randn(d_in, d_out)
        self.W_key = np.random.randn(d_in, d_out)
        self.W_value = np.random.randn(d_in, d_out)
        self.dropout_rate = dropout
        self.mask = np.triu(np.ones((context_length, context_length), dtype=bool), k=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape # batch, num_tokens, input dimension
        keys = np.matmul(x, self.W_key)
        queries = np.matmul(x, self.W_query)
        values = np.matmul(x, self.W_value)

        attn_scores = np.matmul(queries, keys.transpose(0, 2, 1))
        attn_scores[:, self.mask] = -np.inf  # Apply masking for each sample in the batch
        attn_weights = softmax(attn_scores / np.sqrt(keys.shape[-1]))

        context_vecs = np.matmul(attn_weights, values)
        return context_vecs



class MultiHeadAttention_np:
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = np.random.randn(d_in, d_out)
        self.W_key = np.random.randn(d_in, d_out)
        self.W_value = np.random.randn(d_in, d_out)
        self.dropout_rate = dropout
        self.mask = np.triu(np.ones((context_length, context_length), dtype=bool), k=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = np.matmul(x, self.W_key)  # Shape: (b, num_tokens, d_out)
        queries = np.matmul(x, self.W_query)
        values = np.matmul(x, self.W_value)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = np.reshape(keys, (b, num_tokens, self.num_heads, self.head_dim))
        values = np.reshape(values, (b, num_tokens, self.num_heads, self.head_dim))
        queries = np.reshape(queries, (b, num_tokens, self.num_heads, self.head_dim))

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = np.transpose(keys, (0, 2, 1, 3))
        queries = np.transpose(queries, (0, 2, 1, 3))
        values = np.transpose(values, (0, 2, 1, 3))

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = np.matmul(queries, np.transpose(keys, (0, 1, 3, 2)))  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores[:, :, mask_bool] = -np.inf

        attn_weights = softmax(attn_scores / np.sqrt(keys.shape[-1]**0.5))
        attn_weights = np.nan_to_num(attn_weights)  # replace nan with zero
        attn_weights = attn_weights * (1 - self.dropout_rate)  # apply dropout

        # Shape: (b, num_heads, num_tokens, head_dim)
        context_vec = np.matmul(attn_weights, values)
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = np.transpose(context_vec, (0, 2, 1, 3)).reshape(b, num_tokens, self.d_out)

        return context_vec
