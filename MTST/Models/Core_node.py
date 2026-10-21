
# Non utilisé dans le main !!!!! 

import torch 
import torch.nn as nn 
from MTST.Processing.encoding_process import ValueEncoder, PositionalEncoder
from MTST.Models.transformer_basis import  TransformerBlock


class MTST(nn.Module):
  
    def __init__(self, input_dim, embed_dim=64, heads=10, dropout=0.1, n_layers=5, output_len=24, max_len=5000):
        super().__init__()
        self.value_encoder = ValueEncoder(input_dim, embed_dim)
        self.pe_high = PositionalEncoder(embed_dim, max_len)
        self.pe_mid = PositionalEncoder(embed_dim, max_len)
        self.pe_low = PositionalEncoder(embed_dim, max_len)
        self.output_len = output_len
        self.input_dim = input_dim
        self.transformer_high_res = TransformerBlock(embed_dim, heads, dropout, n_layers)
        self.transformer_mid_res = TransformerBlock(embed_dim, heads, dropout, n_layers)
        self.transformer_low_res = TransformerBlock(embed_dim, heads, dropout, n_layers)
        self.fusion_layer = nn.Linear(embed_dim * 3, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_len * 1)
        self.relu = nn.ReLU()
    
    def interpolate(self, x, target_size):
        
        x = x.permute(0, 2, 1)  # (B, embed_dim, seq_len)
        x = nn.functional.interpolate(x, size=target_size, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (B, target_size, embed_dim)
        return x

    
    def forward(self, x, high_res, mid_res, low_res):
        
        x_high = x[:, ::high_res, :]
        x_mid = x[:, ::mid_res, :]
        x_low = x[:, ::low_res, :]

        def process(xr, transformer, positional_encoder):
           
            xr = self.value_encoder(xr)
            xr = positional_encoder(xr)
            xr = transformer(xr)
            return xr
        
        x_high = process(x_high, self.transformer_high_res, self.pe_high)
        x_mid = self.interpolate(process(x_mid, self.transformer_mid_res, self.pe_mid), x_high.size(1))
        x_low = self.interpolate(process(x_low, self.transformer_low_res, self.pe_low), x_high.size(1))

        fused = torch.cat((x_high, x_mid, x_low), dim=-1)
        fused = self.fusion_layer(fused)
        fused = self.linear1(fused)
        fused = torch.relu(fused)
        context = fused.mean(dim=1)
        output = self.linear2(context) 
        forecast = output.view(-1, self.output_len)
        return forecast