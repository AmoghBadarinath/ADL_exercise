import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
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
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        # we need softmax layer and dropout
        self.softmax = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(dropout)
        # as well as the q linear layer
        self.query_repr = nn.Linear(dim, inner_dim, bias = False)

        # and the k/v linear layer (can be realized as one single linear layer
        # or as two individual ones)
        self.key_repr = nn.Linear(dim, inner_dim, bias = False)
        self.value_repr = nn.Linear(dim, inner_dim, bias = False)
        # and the output linear layer followed by dropout
        self.out_repr = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention 
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        b, n, _, h = *x.shape, self.heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim = 1) 
        
        # Linear projections to obtain Q, K, and V from input/context
        q = self.query_repr(x)
        k = self.key_repr(context)
        v = self.value_repr(context)

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        # Compute scaled dot-product attention: (Q × Kᵀ) / √d
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.softmax(attn)
        # Save attention weights
        self.attn_weights = attn
        # Dropout for regularizing attention weights
        attn = self.attn_dropout(attn)
        # Weighted sum of values: Attention weights × V
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_repr(out)

        return out 



# ViT & CrossViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """
    def __init__(self, dim_outer, dim_inner, fn):
        """
        Args:
            dim_outer (int): Input (and output) dimension.
            dim_inner (int): Intermediate dimension (expected by fn).
            fn (callable): A callable object (like a layer).
        """
        super().__init__()
        self.fn = fn
        need_projection = dim_outer != dim_inner
        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input tensor
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape   
            - fn(W_in) * W_out
        """
        # Project input to inner dimension
        x = self.project_in(x)
        
        # Apply the function
        x = self.fn(x, *args, **kwargs)
        
        # Project back to outer dimension
        x = self.project_out(x)
        
        return x

# CrossViT
# cross attention transformer
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        # Create depth number of cross attention layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Small tokens attending to large patches
                ProjectInOut(
                    dim_outer=sm_dim,
                    dim_inner=lg_dim,
                    fn=PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))
                ),
                # Large tokens attending to small patches
                ProjectInOut(
                    dim_outer=lg_dim,
                    dim_inner=sm_dim,
                    fn=PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout))
                )
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # Split CLS tokens and patch tokens
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        # Forward pass through cross attention layers
        for sm_to_lg, lg_to_sm in self.layers:
            # Small CLS token attending to large patches
            sm_cls = sm_to_lg(sm_cls, context=lg_patch_tokens, kv_include_self=False) + sm_cls
            # Large CLS token attending to small patches
            lg_cls = lg_to_sm(lg_cls, context=sm_patch_tokens, kv_include_self=False) + lg_cls

        # Concatenate CLS tokens back with patch tokens
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)

        return sm_tokens, lg_tokens

# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        # Create multiple encoder layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Small patch transformer
                Transformer(dim=sm_dim, **sm_enc_params),
                
                # Large patch transformer
                Transformer(dim=lg_dim, **lg_enc_params),
                
                # Cross attention between small and large patches
                CrossTransformer(
                    sm_dim=sm_dim,
                    lg_dim=lg_dim,
                    depth=cross_attn_depth,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                    dropout=dropout
                )
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # Forward through each layer
        for sm_enc, lg_enc, cross_attend in self.layers:
            # Process tokens through their respective transformers
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)
            
            # Cross attention between small and large representations
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # Create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Create/initialize positional embedding (will be learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Create cls token (for each patch embedding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Create dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # Forward through patch embedding layer
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Concat class tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x += self.pos_embedding[:, :(n + 1)]

        # Apply dropout and return
        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def get_last_attention_weights(self):
        """
        Retrieves the attention weights from the last self-attention block
        of the final Transformer layer.
        """
        # Access the last layer in the transformer stack
        last_transformer_layer = self.transformer.layers[-1]
        
        # The first element of the layer is the PreNorm(Attention) module
        pre_norm_attn = last_transformer_layer[0].fn
        
        # The attention weights are stored in the Attention module itself
        weights = pre_norm_attn.attn_weights
        
        # Clear the weights to prevent leakage or accidental use later
        pre_norm_attn.attn_weights = None 
        
        return weights # Shape: (B, H, N_tokens, N_tokens)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        self.sm_embedder = ImageEmbedder(
            dim=sm_dim,
            image_size=image_size,
            patch_size=sm_patch_size,
            dropout=emb_dropout
        )
        self.lg_embedder = ImageEmbedder(
            dim=lg_dim,
            image_size=image_size,
            patch_size=lg_patch_size,
            dropout=emb_dropout
        )

        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # apply image embedders
        sm_tokens = self.sm_embedder(img)
        lg_tokens = self.lg_embedder(img)

        # and the multi-scale encoder
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # call the mlp heads w. the class tokens 
        sm_cls_token = sm_tokens[:, 0]
        lg_cls_token = lg_tokens[:, 0]
        sm_logits = self.sm_mlp_head(sm_cls_token)
        lg_logits = self.lg_mlp_head(lg_cls_token)
        
        return sm_logits + lg_logits


    def get_last_attention_weights(self):
        """
        Retrieves the attention weights from the last self-attention block 
        of the final SMALL-PATCH Transformer layer within the MultiScaleEncoder.
        """
        # 1. Access the last layer in the MultiScaleEncoder stack
        last_ms_layer = self.multi_scale_encoder.layers[-1]
        
        # 2. The first element is the small patch Transformer (sm_enc)
        sm_transformer = last_ms_layer[0] 
        
        # 3. Access the last layer in the small patch Transformer stack
        last_sm_enc_layer = sm_transformer.layers[-1]
        
        # 4. The first element of that layer is the PreNorm(Attention) module
        pre_norm_attn = last_sm_enc_layer[0].fn
        
        # 5. The Attention weights are stored in the Attention module itself
        weights = pre_norm_attn.attn_weights
        
        # Clear the weights to prevent leakage or accidental use later
        # NOTE: This must be done on the Attention instance which is PreNorm(Attention).fn
        pre_norm_attn.attn_weights = None 
        
        if weights is None:
            raise AttributeError("Attention weights were not computed or saved during the last forward pass.")
            
        return weights # Shape: (B, H, N_tokens, N_tokens)

if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1, emb_dropout = 0.1)
    cvit = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, lg_dim = 128, sm_patch_size = 8,
                    sm_enc_depth = 2, sm_enc_heads = 8, sm_enc_mlp_dim = 128, sm_enc_dim_head = 64,
                    lg_patch_size = 16, lg_enc_depth = 2, lg_enc_heads = 8, lg_enc_mlp_dim = 128,
                    lg_enc_dim_head = 64, cross_attn_depth = 2, cross_attn_heads = 8, cross_attn_dim_head = 64,
                    depth = 3, dropout = 0.1, emb_dropout = 0.1)
    print(vit(x).shape)
    print(cvit(x).shape)
