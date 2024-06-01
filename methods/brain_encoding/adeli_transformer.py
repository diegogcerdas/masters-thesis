import torch
import torch.nn as nn
import open_clip
import numpy as np
import functools

class Backbone_dino(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc_output_layer = list(range(12))

        # Load DINO model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768

        # Disable gradient computation
        for _, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        return self.backbone.get_intermediate_layers(x, n=self.enc_output_layer)
    

class CLIPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, _, self.transform = open_clip.create_model_and_transforms(
            model_name="ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.transform.transforms.pop(-3)  # remove rgb transform
        self.transform.transforms.pop(-2)  # remove totensor transform
        self.num_channels = 1280

        self.enc_output_layer = np.linspace(4, 32, 8).astype(int) - 1

        self.interm_feats = {}
        for enc_output_layer in self.enc_output_layer:
          custom_f = functools.partial(self.hook_fn_forward_interm_feats, enc_output_layer)
          self.clip.visual.transformer.resblocks[enc_output_layer].ls_2.register_forward_hook(custom_f)

    def hook_fn_forward_interm_feats(self, enc_output_layer, module, input, output):
        output = self.clip.visual.ln_post(output.permute(1,0,2))
        self.interm_feats[enc_output_layer] = output[:,1:]

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self.transform(x)
        _ = self.clip.encode_image(x)
        out = [self.interm_feats[k] for k in sorted(self.interm_feats.keys())]
        return out
    

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, num_classes, num_patches, dropout=0.0):
        super().__init__()

        # Layers/Networks
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        # x = x + self.pos_embedding

        # Apply Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    

class DETR_Brain_Encoder(nn.Module):

    def __init__(self, output_size):
        super().__init__()

        self.backbone = Backbone_dino()
        feature_dim = self.backbone.num_channels

        self.transformers = nn.ModuleList([
            VisionTransformer(
                embed_dim=feature_dim,
                hidden_dim=512,
                num_heads=6,
                num_layers=1,
                num_classes=output_size,
                num_patches=256,
                dropout=0.25,
            ) for _ in range(12)
        ])

        self.aggregator = nn.Parameter(torch.randn(1, 12, output_size))


    def forward(self, x):
        with torch.no_grad():
            features  = self.backbone(x)
        out = torch.stack([transformer(features[i]) for (i, transformer) in enumerate(self.transformers)], dim=1).to(x.device)
        out = (out * self.aggregator).sum(dim=1)
        return out
    