import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ViT & CrossViT
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ViT & CrossViT
# Attention class for multi-head self-attention
class Attention(nn.Module):
    # Added save_weights argument
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., save_weights=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.save_weights = save_weights
        self.attn_weights = None # For attention weight visualization

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        # Save weights if requested
        if self.save_weights:
            self.attn_weights = attn.detach().cpu()
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ViT
# Transformer Encoder Block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            # Pass save_weights=True only to the LAST attention layer
            is_last_layer = (i == depth - 1)
            self.layers.append(nn.ModuleList([
                # CRITICAL FIX: Pass 'is_last_layer' to enable saving in the last block's attention module
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, save_weights=is_last_layer)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
            
    # --- ADDED: Method to control attention saving ---
    def set_save_attention_weights(self, value: bool):
        """ Recursively enable/disable saving attention weights on the last layer's Attention module. """
        # layers[-1] is the last block, [0] is PreNorm, .fn is the Attention module
        self.layers[-1][0].fn.save_weights = value
    # --------------------------------------------------
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.3, emb_dropout = 0.3):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean', 'none'}, 'pool type must be either cls (class token), mean (mean pooling) or none (encoder mode)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if self.pool in {'cls', 'mean'}:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        else:
            self.mlp_head = nn.Identity()

    # --- ADDED: Attention weight control methods ---
    def set_save_attention_weights(self, value: bool):
        """ Pass the call down to the transformer """
        self.transformer.set_save_attention_weights(value)

    def get_last_attention_weights(self):
        """ Retrieve weights from the last attention block of the self-attention stack """
        weights = self.transformer.layers[-1][0].fn.attn_weights
        if weights is None:
            raise AttributeError("Attention weights not saved. Did you call `set_save_attention_weights(True)` before the forward pass?")
        
        # Clear the weights after retrieval
        self.transformer.layers[-1][0].fn.attn_weights = None
        return weights
    # --------------------------------------------------

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool == 'none':
            return x
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# CrossViT
# This block performs cross-attention between two different sized tokens
class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_v_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == q_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(q_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(k_v_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, q_dim), 
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, context):
        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)
        k, v = kv

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# CrossViT
# Transformer Encoder for CrossViT
class TwoStreamTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, mlp_dim, dropout = 0., **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(sm_dim, CrossAttention(q_dim=sm_dim, k_v_dim=lg_dim, heads=heads, dim_head=sm_dim // heads, dropout=dropout)),
                PreNorm(lg_dim, CrossAttention(q_dim=lg_dim, k_v_dim=sm_dim, heads=heads, dim_head=lg_dim // heads, dropout=dropout)),
                
                PreNorm(sm_dim, FeedForward(sm_dim, mlp_dim, dropout = dropout)),
                PreNorm(lg_dim, FeedForward(lg_dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_attn, lg_attn, sm_ff, lg_ff in self.layers:
            sm_tokens = sm_attn(sm_tokens, context=lg_tokens) + sm_tokens
            lg_tokens = lg_attn(lg_tokens, context=sm_tokens) + lg_tokens
            
            sm_tokens = sm_ff(sm_tokens) + sm_tokens
            lg_tokens = lg_ff(lg_tokens) + lg_tokens
        return sm_tokens, lg_tokens

# CrossViT
class CrossViT(nn.Module):
    def __init__(self, *, image_size, num_classes, sm_dim, lg_dim, sm_patch_size = 8, sm_enc_depth = 4, sm_enc_heads = 8, sm_mlp_dim = 128, lg_patch_size = 16, lg_enc_depth = 4, lg_enc_heads = 8, lg_mlp_dim = 256, cross_attn_depth = 2, cross_attn_heads = 8, pool = 'cls', channels = 3, dropout = 0.3, emb_dropout = 0.3):
        super().__init__()
        
        # --- Small Patches Stream (Sm ViT) ---
        # Note: We use the ViT class with pool='none' for the internal encoders
        self.sm_vit = ViT(
            image_size = image_size,
            patch_size = sm_patch_size,
            num_classes = num_classes,
            dim = sm_dim,
            depth = sm_enc_depth,
            heads = sm_enc_heads,
            mlp_dim = sm_mlp_dim,
            pool = 'none', 
            channels = channels,
            dropout = dropout,
            emb_dropout = emb_dropout
        )

        # --- Large Patches Stream (Lg ViT) ---
        self.lg_vit = ViT(
            image_size = image_size,
            patch_size = lg_patch_size,
            num_classes = num_classes,
            dim = lg_dim,
            depth = lg_enc_depth,
            heads = lg_enc_heads,
            mlp_dim = lg_mlp_dim,
            pool = 'none', 
            channels = channels,
            dropout = dropout,
            emb_dropout = emb_dropout
        )
        
        # --- Cross-Attention Module ---
        self.cross_attend = TwoStreamTransformer(
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            depth = cross_attn_depth,
            heads = cross_attn_heads,
            mlp_dim = max(sm_mlp_dim, lg_mlp_dim),
            dropout = dropout
        )
        
        # --- Final Classification Head (after pooling) ---
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(sm_dim + lg_dim),
            nn.Linear(sm_dim + lg_dim, num_classes)
        )

    # --- ADDED: Attention weight control methods ---
    def set_save_attention_weights(self, value: bool):
        """ Delegates saving control to the small patch encoder (sm_vit). """
        # We visualize the standard self-attention in the smallest patch encoder.
        self.sm_vit.set_save_attention_weights(value)

    def get_last_attention_weights(self):
        """ Retrieves weights from the last self-attention block of the SMALL patch encoder. """
        return self.sm_vit.get_last_attention_weights()
    # ------------------------------------------------

    def forward(self, img):
        # 1. Forward pass through two independent ViT encoders
        sm_tokens_plus_cls = self.sm_vit(img)
        lg_tokens_plus_cls = self.lg_vit(img)

        sm_cls, sm_tokens = sm_tokens_plus_cls[:, 0:1], sm_tokens_plus_cls[:, 1:]
        lg_cls, lg_tokens = lg_tokens_plus_cls[:, 0:1], lg_tokens_plus_cls[:, 1:]
        
        # 2. Cross-Attention
        sm_tokens, lg_tokens = self.cross_attend(sm_tokens, lg_tokens)

        # 3. Concatenate the CLS token back to the patch tokens
        sm_tokens = torch.cat((sm_cls, sm_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_tokens), dim = 1)

        # 4. Final Pooling
        sm_final = sm_tokens.mean(dim = 1) if self.pool == 'mean' else sm_tokens[:, 0]
        lg_final = lg_tokens.mean(dim = 1) if self.pool == 'mean' else lg_tokens[:, 0]

        # 5. Concatenate and Classify
        combined_features = torch.cat((sm_final, lg_final), dim = -1)
        
        return self.mlp_head(combined_features)