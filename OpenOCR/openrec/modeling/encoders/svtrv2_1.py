import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_
import torch.utils.checkpoint as checkpoint

from openrec.modeling.common import DropPath, Identity, Mlp
from openrec.modeling.encoders.focalsvtr import BasicLayer

from openrec.modeling.encoders.gc_vit import GCViTLayer


class GlobalChannelAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        query = key = self.GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)
        
        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, ):
        super().__init__()

        assert (in_channels % 2 == 0), "in_channel size must be even"

        num_reduced_channels = in_channels // 2
        
        self.conv1x1_q = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_k = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_v = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_att = nn.Conv2d(num_reduced_channels, in_channels, 1, 1)
        
    def forward(self, feature_maps, global_channel_output):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.reshape(N, C, -1)
        key = self.conv1x1_k(feature_maps).reshape(N, C, -1)
        
        query_key = torch.bmm(key.permute(0, 2, 1), query)
        query_key = query_key.reshape(N, -1).softmax(-1)
        query_key = query_key.reshape(N, int(H*W), int(H*W))
        value = self.conv1x1_v(feature_maps).reshape(N, C, -1)
        att = torch.bmm(value, query_key).reshape(N, C, H, W)
        att = self.conv1x1_att(att)
        
        return (global_channel_output * att) + global_channel_output

class LocalChannelAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        N, C, H, W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att =  att.reshape(N, C, 1, 1)
        return (x * att) + x

class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        assert (in_channels % 2 == 0), "in_channel size must be even"

        num_reduced_channels = in_channels // 2
        
        self.conv1x1_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_2 = nn.Conv2d(int(num_reduced_channels * 4), 1, 1, 1)
        
        self.dilated_conv3x3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=2, dilation=2)
        self.dilated_conv7x7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=3, dilation=3)
        
    def forward(self, feature_maps, local_channel_output):
        att = self.conv1x1_1(feature_maps) # b, 16, h, w
        d1 = self.dilated_conv3x3(att)     
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        att = torch.cat((att, d1, d2, d3), dim=1)   # b, 16*4, h, w
        att = self.conv1x1_2(att)   # b, 16*4, h, w
        return (local_channel_output * att) + local_channel_output # (b, c, h, w) * b, 16*4, h, w
    
class GLAM(nn.Module):
    
    def __init__(self, in_channels, kernel_size):     
        
        super().__init__()
        
        self.local_channel_att = LocalChannelAttention(kernel_size)
        self.local_spatial_att = LocalSpatialAttention(in_channels)
        self.global_channel_att = GlobalChannelAttention(kernel_size)
        self.global_spatial_att = GlobalSpatialAttention(in_channels)
        
        self.fusion_weights = nn.Parameter(torch.Tensor([0.333, 0.333, 0.333])) 
        
    def forward(self, x):
        # x shape = [b, c, h, w]
        local_channel_att = self.local_channel_att(x) 
        local_att = self.local_spatial_att(x, local_channel_att) 
        global_channel_att = self.global_channel_att(x) 
        global_att = self.global_spatial_att(x, global_channel_att) 
        
        local_att = local_att.unsqueeze(1) 
        global_att = global_att.unsqueeze(1) 
        x = x.unsqueeze(1) 
        
        all_feature_maps = torch.cat((local_att, x, global_att), dim=1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        fused_feature_maps = (all_feature_maps * weights).sum(1)
        
        return fused_feature_maps   # [b, c, h, w]
    
def sincos_positional_encoding_flatten(x, temperature=10000.0):
 
    batch_size, seq_len, channels = x.shape    
   
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    position = position.to(x.device)    
   
    div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float) * -(math.log(temperature) / channels))
    div_term = div_term.to(x.device)    
  
    pe = torch.zeros(seq_len, channels, device=x.device)    
    
    pe[:, 0::2] = torch.sin(position * div_term)    
 
    if channels % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:channels//2])    
    
    pe = pe.unsqueeze(0).expand(batch_size, -1, -1)    
  
    return x + pe
    
def sincos_positional_encoding(x, temperature=10000.0):   
    x = x.permute(0, 2, 3, 1) 
   
    batch_size, height, width, channels = x.shape    
   
    channels_half = channels // 2    
   
    y_pos = torch.arange(height, dtype=torch.float, device=x.device).unsqueeze(1)
    x_pos = torch.arange(width, dtype=torch.float, device=x.device).unsqueeze(1)    
   
    div_term_y = torch.exp(torch.arange(0, channels_half, 2, dtype=torch.float, device=x.device) * 
                         -(math.log(temperature) / channels_half))
    div_term_x = torch.exp(torch.arange(0, channels_half, 2, dtype=torch.float, device=x.device) * 
                         -(math.log(temperature) / channels_half))
    
   
    pe_y = torch.zeros(height, channels_half, device=x.device)
    pe_y[:, 0::2] = torch.sin(y_pos * div_term_y)
    if channels_half % 2 == 0:
        pe_y[:, 1::2] = torch.cos(y_pos * div_term_y)
    else:
        pe_y[:, 1::2] = torch.cos(y_pos * div_term_y[:channels_half//2])    
   
    pe_x = torch.zeros(width, channels_half, device=x.device)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term_x)
    if channels_half % 2 == 0:
        pe_x[:, 1::2] = torch.cos(x_pos * div_term_x)
    else:
        pe_x[:, 1::2] = torch.cos(x_pos * div_term_x[:channels_half//2])    
   
    pe = torch.zeros(height, width, channels, device=x.device)    
    
    pe_y = pe_y.unsqueeze(1).expand(-1, width, -1)  # [height, width, channels_half]
    pe_x = pe_x.unsqueeze(0).expand(height, -1, -1)  # [height, width, channels_half]    
   
    pe[..., :channels_half] = pe_y
    pe[..., channels_half:2*channels_half] = pe_x    
  
    if channels > 2*channels_half:      
        pe[..., -1] = pe_y[..., 0]    
   
    pe = pe.unsqueeze(0).expand(batch_size, -1, -1, -1)    

    x = x + pe  # [b, h, w, c]
    x = x.permute(0, 3, 1, 2) 
  
    return x 

class ConvBNLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ChannelMultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads = 8,            
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            ):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = dim
        self.head_dim = dim // num_heads   
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = qk_scale or self.head_dim**-0.5

    def forward(self, x):
        # x: (B, C, N) where N=H*W
        
        B, C, N = x.shape            
        x_heads = x.reshape(B, self.num_heads, self.head_dim, N)  # (B, num_heads, head_dim, N)        
    
        Q = x_heads  # (B, num_heads, head_dim, N)
        K = x_heads  # (B, num_heads, head_dim, N)
        V = x_heads  # (B, num_heads, head_dim, N)        
    
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, head_dim, head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, V)  # (B, num_heads, head_dim, N)        
       
        x = x.reshape(B, C, N)  # (B, C, N)
        x = self.proj(x.transpose(1,2)).transpose(1,2)
        x = self.proj_drop(x)
        
        return x    # (B, C, N)

class FeatureFusion(nn.Module):
    def __init__(
            self, 
            dim,
            num_heads,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,            
            reduction=16,
            norm_layer=nn.LayerNorm,
            eps=1e-6,
            mlp_ratio=4.0,     
            act_layer=nn.GELU,
            ):
        super(FeatureFusion, self).__init__()

        self.position_att = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )    

        self.channel_att = ChannelMultiHeadSelfAttention(
            dim,
            num_heads=num_heads,                 
            attn_drop=attn_drop,
            proj_drop=drop,   
            qk_scale=None,        
        )

        self.in_channels = dim
        reduced_channels = dim // reduction

        # Local context (L(z)):
        self.conv1 = nn.Conv2d(dim, reduced_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.conv2 = nn.Conv2d(reduced_channels, dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim)

        # Global context (G(z)):
        # Sử dụng adaptive avg pooling rồi các lớp fc (ở đây cũng được cài đặt dưới dạng conv 1x1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, reduced_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(reduced_channels)
        self.fc2 = nn.Conv2d(reduced_channels, dim, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

        self.norm1 = norm_layer(dim, eps=eps)     
        self.norm2 = norm_layer(dim, eps=eps)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm3 = norm_layer(dim, eps=eps)

    def forward(self, x, sz):
        """
        x (shape: (B, N, C))
        """ 
        B, N, em_dim = x.shape  # N = H * W
        H, W = sz        

        position_attn = self.norm1(x + self.position_att(x))   # b, N, C       
        p = position_attn.transpose(1, 2).reshape(B, em_dim, H, W)

        x_c = x.transpose(1, 2) # b, C, N
        channel_attn = self.norm2(x + self.channel_att(x_c).transpose(1, 2))    # b, N, C        
        c = channel_attn.transpose(1, 2).reshape(B, em_dim, H, W)
       
        z = p + c  # (B, C, H, W)        
        
        # local context L(z)
        Lz = self.conv1(z)
        if B > 1:
            Lz = self.bn1(Lz)
        Lz = self.gelu1(Lz)
        Lz = self.conv2(Lz)
        if B > 1:
            Lz = self.bn2(Lz)   # (B, C, H, W)
        
        # global context G(z)
        gz = self.global_pool(z)  # (B, C, 1, 1)
        Gz = self.fc1(gz)
        if B > 1:
            Gz = self.bn3(Gz)
        Gz = self.gelu2(Gz)
        Gz = self.fc2(Gz)
        if B > 1:
            Gz = self.bn4(Gz)        
        Gz = Gz.expand_as(z) # (B, C, H, W)       
        
        Mz = torch.sigmoid(Lz + Gz)  # (B, C, H, W)      
        
        # Fusion: weighted sum của p và c theo attention weight M(z)
        Ff = Mz * p + (1 - Mz) * c  # (B, C, H, W)

        x = Ff.flatten(2).transpose(1, 2) # [b, N, C] 

        x = self.norm3(x + self.mlp(x))

        return x, sz

class GLAMBlock(nn.Module):

    def __init__(self, 
                dim, 
                kernel_size = 3,
                mlp_ratio=4.0,                
                drop=0.0,                
                drop_path=0.0,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                eps=1e-6,
                ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = GLAM(
            dim,
            kernel_size=kernel_size,            
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, sz):
        if x.ndim == 4: # b, c, h, w
            x = x.flatten(2).transpose(1, 2) # b, N, C   

        if x.ndim == 3: # b, N, C
            C = x.shape[-1]
            x_t = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1]) # b, c, h, w

        x_mixer = self.mixer(x_t) # b, c, h, w
        x_mixer = x_mixer.flatten(2).transpose(1, 2) # b, N, C    

        x = self.norm1(x + self.drop_path(x_mixer))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x    # b, N, C


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class FlattenBlockRe2D(Block):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0,
                 attn_drop=0,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 eps=0.000001):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, eps)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) 

class ConvBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
        num_conv=2,
        kernel_size=3,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(*[
            nn.Conv2d(
                dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads)
            for i in range(num_conv)
        ])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


class FlattenTranspose(nn.Module):

    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class SubSample2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        # print(x.shape)
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x, [H, W]


class SubSample1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class IdentitySize(nn.Module):

    def forward(self, x, sz):
        return x, sz


class SVTRStage(nn.Module):

    def __init__(self,
                 dim=64,
                 out_dim=256,
                 depth=3,
                 mixer=['Local'] * 3,
                 kernel_sizes=[3] * 3,
                 sub_k=[2, 1],
                 num_heads=2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path=[0.1] * 3,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 num_conv=[2] * 3,
                 downsample=None,
                 **kwargs):
        super().__init__()
        self.dim = dim

        self.blocks = nn.Sequential()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(
                    ConvBlock(dim=dim,
                              kernel_size=kernel_sizes[i],
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              drop=drop_rate,
                              act_layer=act,
                              drop_path=drop_path[i],
                              norm_layer=norm_layer,
                              eps=eps,
                              num_conv=num_conv[i]))
                              
            elif mixer[i] == 'GCViT':
                self.blocks.append(
                    GCViTLayer(
                        dim,      
                        num_heads=num_heads,
                        max_window_size=max_window_size,
                        window_size=window_size,                
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        layer_scale=1e-5,
                        max_ratio=max_ratio,
                    )   
                )   
            else:
                if mixer[i] == 'Global':
                    block = Block
                elif mixer[i] == 'FGlobal':
                    block = Block
                    self.blocks.append(FlattenTranspose())
                elif mixer[i] == 'FGlobalRe2D':
                    block = FlattenBlockRe2D
                self.blocks.append(
                    block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                    ))

        if downsample:
            if mixer[-1] == 'Conv' or mixer[-1] == 'FGlobalRe2D':
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        for blk in self.blocks:
            x = blk(x)
        x, sz = self.downsample(x, sz)
        return x, sz


class ADDPosEmbed(nn.Module):

    def __init__(self, feat_max_size=[8, 32], embed_dim=768):
        super().__init__()
        pos_embed = torch.zeros(
            [1, feat_max_size[0] * feat_max_size[1], embed_dim],
            dtype=torch.float32)
        trunc_normal_(pos_embed, mean=0, std=0.02)
        self.pos_embed = nn.Parameter(
            pos_embed.transpose(1, 2).reshape(1, embed_dim, feat_max_size[0],
                                              feat_max_size[1]),
            requires_grad=True,
        )

    def forward(self, x):
        sz = x.shape[2:]
        x = x + self.pos_embed[:, :, :sz[0], :sz[1]]
        return x


class POPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 in_channels=3,
                 feat_max_size=[8, 32],
                 embed_dim=768,
                 use_pos_embed=False,
                 flatten=False,
                 bias=False):
        super().__init__()
        self.patch_embed = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=bias,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=bias,
            ),
        )
        if use_pos_embed:
            self.patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            self.patch_embed.append(FlattenTranspose())

    def forward(self, x):
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]


class LastStage(nn.Module):

    def __init__(self, in_channels, out_channels, last_drop, out_char_num=0):
        super().__init__()
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x, sz):
        x = x.reshape(-1, sz[0], sz[1], x.shape[-1])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class Feat2D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz


class SVTRv2_1(nn.Module):

    def __init__(self,
                 max_sz=[32, 128],
                 in_channels=3,
                 out_channels=192,
                 depths=[3, 6, 3],    
                 depths_focal=6,
                 focal_levels=3, 
                 focal_windows=3,
                 use_conv_embed=False, 
                 use_checkpoint=False,  
                 use_layerscale=False, 
                 layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 dims=[64, 128, 256],
                 mixer=[['Conv'] * 3, ['Conv'] * 3 + ['Global'] * 3,
                        ['Global'] * 3],
                 use_pos_embed=True,
                 sub_k=[[1, 1], [2, 1], [1, 1]],
                 num_heads=[2, 4, 8],
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 last_drop=0.1,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 last_stage=False,
                 feat2d=False,
                 eps=1e-6,
                 num_convs=[[2] * 3, [2] * 3 + [3] * 3, [3] * 3],
                 kernel_sizes=[[3] * 3, [3] * 3 + [3] * 3, [3] * 3],
                 pope_bias=False,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(in_channels=in_channels,
                                 feat_max_size=feat_max_size,
                                 embed_dim=dims[0],
                                 use_pos_embed=use_pos_embed,
                                 flatten=mixer[0][0] != 'Conv',
                                 bias=pope_bias)

        dpr = np.linspace(0, drop_path_rate,
                          sum(depths))  # stochastic depth decay rule  
       
        # build layers       
        self.focal_layer = BasicLayer(
            dim=dims[0],
            out_dim=None,
            input_resolution=None,
            depth=depths_focal,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=0.1,
            norm_layer=norm_layer,
            downsample=None,
            downsample_kernel=sub_k[0],
            focal_level=focal_levels,
            focal_window=focal_windows,
            use_conv_embed=use_conv_embed,
            use_checkpoint=use_checkpoint,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            use_postln=use_postln,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )  

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=kernel_sizes[i_stage]
                if len(kernel_sizes[i_stage]) == len(mixer[i_stage]) else [3] *
                len(mixer[i_stage]),
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
                num_conv=num_convs[i_stage] if len(num_convs[i_stage]) == len(
                    mixer[i_stage]) else [2] * len(mixer[i_stage]),
            )
            self.stages.append(stage)    

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(
                LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            self.stages.append(Feat2D())
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        x, sz = self.pope(x)
        
        x = x.flatten(2).transpose(1, 2)
        H = sz[0]
        W = sz[1]       
        x, H, W = self.focal_layer(x, H, W)
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)

        for stage in self.stages:
            x, sz = stage(x, sz)

        return x
