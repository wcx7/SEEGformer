import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Layer normalization after self-attention
        self.norm1 = nn.LayerNorm(normalized_shape=dim)

        # Feed-Forward Neural Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio*dim, dim), 
            nn.Dropout(dropout)
        )

        # Layer normalization after feed-forward network
        self.norm2 = nn.LayerNorm(normalized_shape=dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output, attn_weights = self.self_attn(x, x, x, need_weights=True, average_attn_weights=True)

        # Residual Connection and Layer Normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-Forward Network
        ffn_output = self.ffn(x)

        # Residual Connection and Layer Normalization
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x, attn_weights
    
if __name__ == '__main__':
    # Example usage of the TransformerBlock
    input_dim = 64
    num_heads = 8
    hidden_dim = 128
    block = TransformerBlock(input_dim, num_heads, hidden_dim)
    input_tensor = torch.rand(32, 113, input_dim)    # Batch size: 32;  Sequence length: 10,channel number of EEG channels;  input_dim:length of the embeded EEG window or slice.
    output_tensor, attn_weights = block(input_tensor)    # torch.Size([Batch size, Sequence length, input_dim])
    output_tensor, attn_weights = block(output_tensor)
    paras = sum([p.data.nelement() for p in block.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))