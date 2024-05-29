import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class Backbone_dino_posembed(nn.Module):

    def __init__(self, enc_output_layer: int):
        super().__init__()

        # Load DINO model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        self.num_heads = 12
        self.patch_size = 14

        # Disable gradient computation
        for _, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)

        # Load positional embedding
        self.pos_embed = PositionEmbeddingLearned(self.num_channels // 2)

        # Save qkv features of specified layer after forward pass
        self.qkv_feats = torch.empty(0)
        self.backbone._modules["blocks"][enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv)

    def hook_fn_forward_qkv(self, module, input, output):
        self.qkv_feats = output

    def forward(self, x: torch.Tensor):
        # Input: (B,C,H,W) with H,W divisible by self.patch_size
        batch_size = x.shape[0]
        num_tokens = (x.shape[-2]//self.patch_size  * x.shape[-1]//self.patch_size) + 1
        # Obtain features from queries
        _ = self.backbone.get_intermediate_layers(x)[0]
        feats = self.qkv_feats
        q = feats.reshape(batch_size, num_tokens, 3, self.num_heads, -1 // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        x = q.transpose(2, 3).reshape(batch_size, -1, num_tokens)[:,:,1:].reshape(batch_size, -1, x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size)
        # Obtain positional embedding
        pos = self.pos_embed(x).to(x.dtype)
        return x, pos
    

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     query_pos: Optional[torch.Tensor] = None):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None,
                    query_pos: Optional[torch.Tensor] = None):
                    
        tgt2 = self.norm1(tgt)
        
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
    

class Transformer(nn.Module):

    def __init__(
        self, 
        d_model=768, 
        nhead=16, 
        num_decoder_layers=1, 
        dim_feedforward=2048, 
        dropout=0.1,
        activation="relu", 
        normalize_before=False,
        return_intermediate_dec=False, 
    ):
        super().__init__()

        assert num_decoder_layers > 0
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self.memory_proj = nn.Conv2d(d_model, 64, kernel_size=1)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = src

        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2) 
    

class DETR_Brain_Encoder(nn.Module):

    def __init__(self, enc_output_layer, output_size):
        super().__init__()

        self.backbone = Backbone_dino_posembed(enc_output_layer)
        feature_dim = self.backbone.num_channels
        self.transformer = Transformer(
            d_model=feature_dim,
            nhead=16,
            num_decoder_layers=1,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.query_embed = nn.Embedding(1, feature_dim)
        self.linear = nn.Linear(feature_dim, output_size)

    def forward(self, x):
        features, pos_embed = self.backbone(x)
        hs = self.transformer(features, self.query_embed.weight, pos_embed) 
        output_tokens = hs[-1]
        pred = self.linear(output_tokens[:,0,:])
        return pred
    