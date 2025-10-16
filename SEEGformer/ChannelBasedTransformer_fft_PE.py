import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F
from utils import get_1d_sincos_pos_embed



class ChannelBasedTransformer_fft(nn.Module):
    def __init__(self, num_heads, fft_dim, embed_dim, mlp_ratio, dropout, n_chans, num_blocks, num_classes, position_encoding, pool = 'mean'):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.n_chans = n_chans
        self.fc_real = nn.Linear(fft_dim,embed_dim)
        self.fc_imag = nn.Linear(fft_dim,embed_dim)
        self.fc_abs = nn.Linear(fft_dim,embed_dim)

        self.fc_position_real = nn.Linear(position_encoding.shape[1], embed_dim)
        self.fc_position_imag = nn.Linear(position_encoding.shape[1], embed_dim)
        self.fc_position_abs = nn.Linear(position_encoding.shape[1], embed_dim)

        self.transformer_blocks_real = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])    # real part
        self.transformer_blocks_imag = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])    # imaginary part
        self.transformer_blocks_abs = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])     # amplitude part
        

        self.pos_embed = nn.Parameter(torch.zeros(1, n_chans + 1, position_encoding.shape[1]), requires_grad=False)  # positional encoding
        
        
        self.cls_token_real = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_imag = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_abs = nn.Parameter(torch.zeros(1, 1, embed_dim))

        
        self.cls_weight_real = nn.Parameter(0.3333 * torch.ones(1,), requires_grad=True)    # classification weight of real part
        self.cls_weight_imag = nn.Parameter(0.3333 * torch.ones(1,), requires_grad=True)    # classification weight of imaginary part
        self.cls_weight_abs = nn.Parameter(0.3333 * torch.ones(1,), requires_grad=True)     # classification weight of amplitude part
       
        self.classifier_real = nn.Linear(embed_dim, num_classes)    # classifier of real part
        self.classifier_imag = nn.Linear(embed_dim, num_classes)    # classifier of imaginary part
        self.classifier_abs = nn.Linear(embed_dim, num_classes)     # classifier of amplitude part

        self.pos_embed.data.copy_(torch.from_numpy(position_encoding).float().unsqueeze(0))


               
    def forward(self, x_real, x_imag, x_abs):
        # real
        x_real = self.fc_real(x_real)
        self.pos_embed_real = self.fc_position_real(self.pos_embed)
        x_real = x_real + self.pos_embed_real[:, 1:, :]     # add with positional encoding
        cls_token_real = self.cls_token_real + self.pos_embed_real[:, :1, :]
        cls_tokens_real = cls_token_real.expand(x_real.shape[0], -1, -1)
        x_real = torch.cat((cls_tokens_real, x_real), dim=1)        
        for index_real, block in enumerate(self.transformer_blocks_real):
            if index_real == len(self.transformer_blocks_real) - 1:
                x_real, attn_weights_real = block(x_real)          
            else:
                x_real, _ = block(x_real)
        x_real = x_real.mean(dim=1) if self.pool == 'mean' else x_real[:, 0]        
        pred_real = self.classifier_real(x_real)

        # imag
        x_imag = self.fc_imag(x_imag)
        self.pos_embed_imag = self.fc_position_imag(self.pos_embed)       
        x_imag = x_imag + self.pos_embed_imag[:, 1:, :]     # add with positional encoding
        cls_token_imag = self.cls_token_imag + self.pos_embed_imag[:, :1, :]
        cls_tokens_imag = cls_token_imag.expand(x_imag.shape[0], -1, -1)
        x_imag = torch.cat((cls_tokens_imag, x_imag), dim=1)        
        for index_imag, block in enumerate(self.transformer_blocks_imag):
            if index_imag == len(self.transformer_blocks_imag) - 1:
                x_imag, attn_weights_imag = block(x_imag)          
            else:
                x_imag, _ = block(x_imag)
        x_imag = x_imag.mean(dim=1) if self.pool == 'mean' else x_imag[:, 0]        
        pred_imag = self.classifier_imag(x_imag)

        # abs
        x_abs = self.fc_abs(x_abs)
        self.pos_embed_abs = self.fc_position_abs(self.pos_embed) 
        x_abs = x_abs + self.pos_embed_abs[:, 1:, :]    # add with positional encoding
        cls_token_abs = self.cls_token_abs + self.pos_embed_abs[:, :1, :]
        cls_tokens_abs = cls_token_abs.expand(x_abs.shape[0], -1, -1)
        x_abs = torch.cat((cls_tokens_abs, x_abs), dim=1)        
        for index_abs, block in enumerate(self.transformer_blocks_abs):
            if index_abs == len(self.transformer_blocks_abs) - 1:
                x_abs, attn_weights_abs = block(x_abs)          
            else:
                x_abs, _ = block(x_abs)
        x_abs = x_abs.mean(dim=1) if self.pool == 'mean' else x_abs[:, 0]        
        pred_abs = self.classifier_abs(x_abs)       

        # weighted sum
        pred = pred_real * self.cls_weight_real +\
               pred_imag * self.cls_weight_imag +\
               pred_abs * self.cls_weight_abs
        out = F.softmax(pred, dim=1)
        return out, attn_weights_real, attn_weights_imag, attn_weights_abs

        
if __name__ == '__main__':
    batch_size = 32
    num_heads = 4
    embed_dim = 64
    mlp_ratio = 2
    dropout = 0.2
    n_chans = 113
    num_blocks = 4
    num_classes = 2
    fft_dim = 100
    pool = 'mean'
    model = ChannelBasedTransformer_fft(num_heads, fft_dim, embed_dim, mlp_ratio, dropout, n_chans, num_blocks, num_classes, pool = 'mean')
    input = torch.rand(batch_size, 113, 100)  # input series
    output,_ = model(input,input,input)
    print(output.shape)
    paras = sum([p.data.nelement() for p in model.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))
