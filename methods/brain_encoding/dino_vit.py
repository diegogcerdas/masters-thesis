import torch
import torch.nn as nn

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
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))


    def forward(self, x):

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        # Apply Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    

class DINO_ViT_Encoder(nn.Module):

    def __init__(self, feature_dim, output_size):
        super().__init__()

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
        out = torch.stack([transformer(x[i]) for (i, transformer) in enumerate(self.transformers)], dim=1).to(x[0].device)
        out = (out * self.aggregator).sum(dim=1)
        return out
    