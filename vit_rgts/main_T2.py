import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn


#utils
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Pos embedding
def pos_emb_sincos_2d(
    h,
    w,
    dim,
    temperature: int = 10000,
    dtype = torch.float32
):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature ** omega)
    
    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


#classes
class FeedForward(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        dropout = 0.
    ):
        """
        Feedforward layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. GELU
        4. Dropout
        5. Linear
        6. Dropout

        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Apply dropout to the input
        3. Apply non-linearity to the input
        4. Apply dropout to the input

        Args:
        -----
        dim: int
            Dimension of input
        hidden_dim: int
            Dimension of hidden layer
        dropout: float
            Dropout rate
        
        Returns:
        --------
        torch.Tensor
            Output of feedforward layer

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #apply feedforward layer to input tensor
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.
    ):
        """
        Attention Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. Rearrange
        4. LayerNorm
        5. Linear
        6. Rearrange
        7. Softmax
        8. Dropout
        9. Rearrange
        10. Linear
        11. Dropout
        
        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Rearrange input tensor
        3. Apply non-linearity to the input
        4. Rearrange input tensor
        5. Apply softmax to the input
        6. Apply dropout to the input
        7. Rearrange input tensor
        8. Apply non-linearity to the input
        
        """
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        #layer norm
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)

        #sftmx
        self.attend = nn.Softmax(dim = -1)

        #dropout
        self.dropout = nn.Dropout(dropout)

        #projections, split from x -> q, k, v
        self.to_qkv = nn.Linear(
            dim, 
            inner_dim * 3, 
            bias = False
        )
        
        #project out
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #apply layernorm to x
        x = self.norm(x)

        #apply linear layer to x
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        #rearrange x to original shape
        q, k, v = map(
            lambda t: rearrange(
                t, 
                'b n (h d) -> b h n d', 
                h = self.heads
            ), qkv)

        # #normalize key and values, known QK Normalization
        k = self.norm_k(k)
        v = self.norm_v(v)
        
        # attn
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            #Flash Attention
            out = F.scaled_dot_product_attention(q, k, v)
            
            #dropout
            out = self.dropout(out)

            #rearrange to original shape
            out = rearrange(out, 'b h n d -> b n (h d)')

            #project out
            return self.to_out(out)
        
        
class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout = 0.
    ):
        """
        Transformer Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Attention
        3. FeedForward

        Args:
        -----
        dim: int
            Dimension of input
        depth: int
            layers of transformers
        heads: int
            Number of heads
        dim_head: int
            Dimension of head
        mlp_dim: int
            Dimension of MLP
        dropout: float
            Dropout rate

        
        """
        super().__init__()
        
        #layer norm
        self.norm = nn.LayerNorm(dim)

        #transformer layers data array
        self.layers = nn.ModuleList([])
        
        #add transformer layers as depth = transformer blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #attention
                Attention(
                    dim, 
                    heads = heads, 
                    dim_head = dim_head, 
                    dropout = dropout
                ),
                #feedforward
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            #layernorm before attention
            x = self.norm(x)
            
            #parallel
            x = x + attn(x) + ff(x)
        
        return self.norm(x)






class VitRGTS(nn.Module):
    """
    ViT with Register Tokens (VitRGTS) - Modified for Coordinate Regression Probing
    """

    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim,
        num_register_tokens: int = 0, 
        pool = 'cls', 
        channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.
    ):
        super().__init__()

        # Patch embedding
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean pooling'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Register tokens
        # Register tokens (optional)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))
        else:
            self.register_tokens = None

        # Positional embedding
        self.pos_embedding = pos_emb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # Transformer backbone
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        # Heads
        self.mlp_head = nn.Linear(dim, num_classes)
        self.linear_head = nn.Linear(dim, num_classes)

        # Save patch and image sizes
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

    def forward(self, img, return_tokens=False):
        """
        Forward pass.

        If return_tokens=True:
            Returns both classification logits and patch token embeddings.
        Otherwise:
            Returns only classification logits (normal training mode).
        """
        batch, device = img.shape[0], img.device

        # 1️⃣ Patch embedding
        x = self.to_patch_embedding(img)

        # 2️⃣ Add positional encoding
        x += self.pos_embedding.to(device)

        if self.register_tokens is not None:
            r = repeat(self.register_tokens, 'n d -> b n d', b=batch)
            x, ps = pack([x, r], 'b * d')
        else:
            ps = [(x.shape[1],)]  # only patches, no register tokens

        # 5️⃣ Pass through Transformer encoder
        x = self.transformer(x)

# 6️⃣ Separate back patch tokens and register tokens (if any)
        if self.register_tokens is not None:
            x, _ = unpack(x, ps, 'b * d')
        else:
    # no registers → x is already just patch tokens
            pass


        # 7️⃣ Global average pooling for classification
        pooled = x.mean(dim=1)
        logits = self.linear_head(self.to_latent(pooled))

        # If we’re probing (for coordinate regression), return both
        if return_tokens:
            return logits, x  # (logits, patch_tokens)
        else:
            return logits
