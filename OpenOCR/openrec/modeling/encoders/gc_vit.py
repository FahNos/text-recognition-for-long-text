import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size_h, window_size_w):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size_h, window_size_w, C)
    return windows  # (num_windows*B, window_size_h, window_size_w, C)


def window_reverse(windows, window_size_h, window_size_w, H, W, C):
    # b*num_win, w_h, w_w, c    

    x = windows.view(-1, H // window_size_h, W // window_size_w, window_size_h, window_size_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)

    return x

class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeatExtract(nn.Module): 

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttention(nn.Module):   

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=[8,32],
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        # window_size = (window_size, window_size)
        self.max_window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5


        max_positions = (2 * self.max_window_size[0] - 1) * (2 * self.max_window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(max_positions, num_heads))       

        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global, window_size):
        B_, N, C = x.shape
        assert N == window_size[0] * window_size[1], "N phải bằng window_size[0] * window_size[1]"

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
       
        relative_coords[:, :, 0] += self.max_window_size[0] - 1
        relative_coords[:, :, 1] += self.max_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.max_window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)


        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            window_size[0] * window_size[1], window_size[0] * window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)


        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionGlobal(nn.Module):  

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=[8,32],
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        # window_size = (window_size, window_size)
        self.max_window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        max_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(max_positions, num_heads))

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global, window_size):
        B_, N, C = x.shape
        assert N == window_size[0] * window_size[1], "N phải bằng window_size[0] * window_size[1]"

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
       
        relative_coords[:, :, 0] += self.max_window_size[0] - 1
        relative_coords[:, :, 1] += self.max_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.max_window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)


        B = q_global.shape[0]
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        B_dim = torch.div(B_, B, rounding_mode='floor')
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
       
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            window_size[0] * window_size[1], window_size[0] * window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)


        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GCViTBlock(nn.Module):  

    def __init__(self,
                 dim,                 
                 num_heads,      
                 window_size=[8,32],         
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0    

    def forward(self, x, q_global, sz):    
        B, _, C = x.shape
        H, W = sz 
        ratio = W // H

        x = x.reshape(B, H, W, C)
       
        x = self.norm1(x)

        if H == 8:
            self.window_size_h = 8
            self.window_size_w = 16
        elif H == 6:
            self.window_size_h = 6
            self.window_size_w = 24
        elif H == 5:
            self.window_size_h = 5
            self.window_size_w = 28
        elif H == 4 and W == 32:
            self.window_size_h = 4
            self.window_size_w = 32  

        elif H == 4 and W > 32:            
            self.window_size_h = 4
            self.window_size_w = 16  
        
        x_windows = window_partition(x, self.window_size_h, self.window_size_w )
        shortcut = x_windows
        
        x_windows = x_windows.view(-1, self.window_size_h * self.window_size_w, C) # b*num_win, w_h, w_w, c
        # print(f'   batch size in GcViT block = {B}')
        # print(f'   batch size after separate window = {x_windows.shape}')

        window_size = [self.window_size_h, self.window_size_w]
        attn_windows = self.attn(x_windows, q_global, window_size) #  [b*, w_h, w_w, c] @ (b, 1, num_head, w_h * w_w, head_dim)
               
        x = attn_windows.reshape(-1, self.window_size_h, self.window_size_w, C)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x_w = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (b*, h, w, C)
       
        # B*num_win, w_h, w_w, c
        x = window_reverse(
            x_w, 
            self.window_size_h,
            self.window_size_w,
            H, 
            W, 
            C
            ) # [b, H, W, C]
        
        x = x.flatten(start_dim=1, end_dim=2)
        
        return x   # [b, N, C]


class GlobalQueryGen(nn.Module):  
    def __init__(self,
                 dim,    
                 num_heads= 8,
                 window_size =[4,16],
                 max_ratio=24):     

        super().__init__()

        log2_max_ratio = math.log2(max_ratio)  
        ceil_log2_max_ratio = math.ceil(log2_max_ratio) 
        self.max_pools = int(ceil_log2_max_ratio) + 1

        self.extracts = nn.ModuleList([
            FeatExtract(dim, keep_dim=False)
            for _ in range(self.max_pools)
        ])       
        
        self.num_heads = num_heads
       
        self.dim_head = dim // num_heads
        self.window_size = window_size
        self.adaptive_pool = nn.AdaptiveMaxPool2d(window_size)

    def forward(self, x, sz):       
        B, _, C = x.shape
        H, W = sz 
        ratio = W // H

        x = x.transpose(1, 2).reshape(B, C, H, W)       

        widths = [W]        
        for i in range(self.max_pools):
            prev = widths[-1]
            next_w = (prev - 1) // 2 + 1
            if next_w < self.window_size[1]:
                break
            widths.append(next_w)
            if next_w % 2 == 1:
                break        
        
        n = len(widths)        
        if ratio > 8:
            for i in range(n):
                x = self.extracts[i](x)
            if widths[-1] != W:       
                x = self.adaptive_pool(x)
        
        x = _to_channel_last(x) # b, h, w, c
        B = x.shape[0]
        x = x.reshape(B, 1, -1, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4) 
        # (b, 1, w_h * w_w, num_head, head_dim) 
        # ==> (b, 1, num_head, w_h * w_w, head_dim)
        #  w_h * w_w = 4 * 8
     
        return x


class GCViTLayer(nn.Module): 

    def __init__(self,
                 dim,          
                 num_heads=8,
                 max_window_size=[8,32],
                 window_size=[4,16],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=1e-5,
                 max_ratio=24):    

        super().__init__()
        
        self.block_1 = GCViTBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=max_window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention=WindowAttention ,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            layer_scale=layer_scale,           
            )
        
        self.block_2 = GCViTBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=max_window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention=WindowAttentionGlobal,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            layer_scale=layer_scale,           
            )
               
        self.q_global_gen = GlobalQueryGen(dim,   
                                           num_heads=num_heads,
                                           window_size=window_size,
                                           max_ratio=max_ratio)

    def forward(self, x, sz): #  x shape = [b, N, C]   

        q_global = self.q_global_gen(x, sz) # (b, 1, num_head, w_h * w_w, head_dim)  
        x = self.block_1(x, q_global, sz)
        x = self.block_2(x, q_global, sz)       

        return x  # [b, N, C]