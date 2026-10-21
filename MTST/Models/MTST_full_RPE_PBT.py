import torch 
import torch.nn as nn 
from MTST.Processing.encoding_process import ValueEncoder, PositionalEncoder
from MTST.Models.transformer_basis import  TransformerBlock, CustomTransformerEncoder


class MTST_full_RPE_PBT(nn.Module):
    
    def __init__(self, input_dim, embed_dim=64, heads=10, dropout=0.1, n_layers=5, output_len=24, max_len=5000):
        
        super().__init__()
        self.value_encoder = ValueEncoder(embed_dim)
        self.pe_high = PositionalEncoder(embed_dim, max_len)
        self.pe_mid = PositionalEncoder(embed_dim, max_len)
        self.pe_low = PositionalEncoder(embed_dim, max_len)
        self.output_len = output_len
        self.input_dim = input_dim

        self.transformer_high_res = CustomTransformerEncoder(embed_dim, heads, dropout, n_layers)
        self.transformer_mid_res  = CustomTransformerEncoder(embed_dim, heads, dropout, n_layers)
        self.transformer_low_res  = CustomTransformerEncoder(embed_dim, heads, dropout, n_layers)

        self.fusion_layer = nn.Linear(embed_dim * 3, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_len * input_dim)
        self.relu = nn.ReLU()

    # PATCH TOKENIZER
    def patch_tokenize(self, x, patch_size, stride):
        B, T, I = x.shape
        J = (T - patch_size) // stride + 1
        patches = [x[:, i*stride:i*stride+patch_size, :].unsqueeze(1) for i in range(J)]
        return torch.cat(patches, dim=1)  # (B, J, P, I)

    #  INTERPOLATION  #
    def interpolate(self, x, target_size):
        x = x.permute(0, 2, 1)
        x = nn.functional.interpolate(x, size=target_size, mode="linear", align_corners=False)
        return x.permute(0, 2, 1)

    # PATCH TOKENIZER
    def patch_tokenize(self, x, patch_size, stride):
        B, T, I = x.shape
        J = (T - patch_size) // stride + 1
        patches = [x[:, i*stride:i*stride+patch_size, :].unsqueeze(1) for i in range(J)]
        return torch.cat(patches, dim=1)  # (B, J, P, I)
    
    #  PROCESSING PIPELINE 
    def process_branch(self, x, patch_size, stride, positional_encoder, transformer):
        # 1. Tokenization
        x_patched = self.patch_tokenize(x, patch_size, stride)  # (B, J, P, I)
        B, J, P, I = x_patched.shape

        # 2. Flatten patches (chaque patch devient un token)
        x_tokens = x_patched.view(B, J, P * I)  # (B, J, P×I)

        # 3. Value encoding
        x_tokens = self.value_encoder(x_tokens)  # (B, J, embed_dim)

        # 4. Position + Transformer (RPE)
        x_tokens = positional_encoder(x_tokens)
        x_tokens = transformer(x_tokens)
        return x_tokens  # (B, J, embed_dim)

    #  FORWARD
    def forward(self, x, high_res, mid_res, low_res):
        # chaque branche a ses paramètres propres
        x_high = self.process_branch(x, patch_size=high_res, stride=high_res//2,
                                     positional_encoder=self.pe_high, transformer=self.transformer_high_res)
        x_mid  = self.process_branch(x, patch_size=mid_res,  stride=mid_res//2,
                                     positional_encoder=self.pe_mid, transformer=self.transformer_mid_res)
        x_low  = self.process_branch(x, patch_size=low_res,  stride=low_res//2,
                                     positional_encoder=self.pe_low, transformer=self.transformer_low_res)

        # Mise à la même longueur
        target_len = x_high.size(1)
        x_mid = self.interpolate(x_mid, target_len)
        x_low = self.interpolate(x_low, target_len)

        # Fusion
        fused = torch.cat((x_high, x_mid, x_low), dim=-1)
        fused = self.fusion_layer(fused)
        fused = self.relu(self.linear1(fused))
        context = fused.mean(dim=1)

        out = self.linear2(context)
        forecast = out.view(-1, self.output_len, self.input_dim)
        return forecast
