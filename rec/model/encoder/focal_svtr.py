import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math

from model.layers import DropPath, Mlp, ConvBNLayer

class FocalModulation(layers.Layer):
    def __init__(
        self,
        dim,
        focal_window,
        focal_level,
        max_kh=None,
        focal_factor=2,
        bias=True,
        proj_drop=0.0,
        use_postln_in_modulation=False,
        normalize_modulator=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.max_kh = max_kh

        self.f = layers.Dense(2 * dim + (self.focal_level + 1), use_bias=bias)
        self.h = layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=bias)

        self.act = layers.Activation('gelu')
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        self.focal_layers = []

        self.kernel_sizes = []
        for k in range(self.focal_level):   # fl=[3,3,3], k=0, ff=2, fw=[3,3,3] => ker=3| k=1 =>ker=5| k=2=>ker=7
            kernel_size = self.focal_factor * k + self.focal_window         
            padding = 'same'
            focal_layer = tf.keras.Sequential([
                layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=1,
                    padding=padding,
                    use_bias=False,
                ),
                layers.Activation('gelu')
            ])
            self.focal_layers.append(focal_layer)
            self.kernel_sizes.append(kernel_size)

        if self.use_postln_in_modulation:
            self.ln = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):     # (b,h,w,c)

        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # pre linear projection
        x_proj = self.f(x)  # (B, H, W, 2*C + focal_level + 1)

        # Split the tensor along the channel dimension
        q = x_proj[:, :, :, :C]
        ctx = x_proj[:, :, :, C:2*C]
        gates = x_proj[:, :, :, 2*C:]   #(B, H, W, focal_level + 1)

        # context aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx_l = self.focal_layers[l](ctx)  # ctx_l = (b,h,w,c)
            gate_l = gates[:, :, :, l:l+1]                        # b,h,w,1
            ctx_all = ctx_all + ctx_l * gate_l                    # ctx_all = (b,h,w,c)

        # Global context
        ctx_global = tf.reduce_mean(ctx, axis=[1, 2], keepdims=True)  #(b,1,1,c)
        ctx_global = self.act(ctx_global)
        ctx_all = ctx_all + ctx_global * gates[:, :, :, self.focal_level:self.focal_level+1]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        modulator = self.h(ctx_all)
        x_out = q * modulator

        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out, training=training)  # (b,h,w,c)

        return x_out


class FocalNetBlock(layers.Layer):
    def __init__(self,
                 dim,
                 input_resolution=None,
                 mlp_ratio=4.0,
                 drop=0.0,
                 drop_path=0.0,
                 focal_level=1,
                 focal_window=3,
                 max_kh=None,  
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.modulation = FocalModulation(
            dim=dim,
            proj_drop=drop,
            focal_window=focal_window,
            focal_level=self.focal_level,
            max_kh=max_kh,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else layers.Lambda(lambda x: x)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0     

    def call(self, x, H=None, W=None, training=False):

        shape = tf.shape(x)
        B, L, C = shape[0], shape[1], shape[2]
        shortcut = x

        # Apply norm before or after modulation based on use_postln flag
        if not self.use_postln:
            x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])
        x = self.modulation(x, training=training)
        x = tf.reshape(x, [B, H * W, C])
        if self.use_postln:
            x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x, training=training)
        if self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), training=training),training=training)
        else:
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x, training=training)),training=training)

        return x

class BasicLayer(layers.Layer):
    def __init__(self,
                 dim,
                 out_dim,
                 input_resolution,
                 depth,
                 mlp_ratio=4.0,
                 drop=0.0,
                 drop_path=0.0,
                 norm_layer=layers.LayerNormalization,
                 downsample=None,
                 downsample_kernel=[],               
                 focal_level=1,
                 focal_window=1,                  
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth        

        # build blocks
        self.blocks = []
        for i in range(depth):
            drop_path_val = drop_path[i] if isinstance(drop_path, list) else drop_path
            block = FocalNetBlock(dim=dim,
                                  input_resolution=input_resolution,
                                  mlp_ratio=mlp_ratio,
                                  drop=drop,
                                  drop_path=drop_path_val,
                                  focal_level=focal_level,
                                  focal_window=focal_window, 
                                  use_postln=use_postln,
                                  use_postln_in_modulation=use_postln_in_modulation,
                                  normalize_modulator=normalize_modulator)
            self.blocks.append(block)

        # downsample layer
        if downsample is not None:           
            self.downsample = downsample(img_size=input_resolution,
                                         patch_size=downsample_kernel,                                       
                                         embed_dim=out_dim,                                         
                                         norm_layer=norm_layer,
                                         is_stem=False)
        else:        
            self.downsample = None

    def call(self, x, H, W, training=False):
        for block in self.blocks:
            x = block(x, H=H, W=W, training=training)

        if self.downsample is not None:
            x = tf.reshape(x, [tf.shape(x)[0], H, W, self.dim])
            x, H, W = self.downsample(x)        

        return x, H, W

class PatchEmbed(layers.Layer):
    def __init__(self,
                 img_size=(224, 224),
                 patch_size=[4, 4],                
                 embed_dim=96,                
                 norm_layer=None,
                 is_stem=False,
                 **kwargs):

        super().__init__(**kwargs)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]       
        self.embed_dim = embed_dim
      
        self.conv_proj = layers.Conv2D(filters=embed_dim,
                                        kernel_size=patch_size,
                                        strides=patch_size,
                                        padding='valid')

        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5)
        else:
            self.norm = None

    def call(self, x, training=False): # (B, H, W, C)
        x = self.conv_proj(x)
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        # Flatten spatial dimensions and transpose to (B, L, C)
        x = tf.reshape(x, [B, H * W, C])

        if self.norm is not None:
            x = self.norm(x, training=training)

        return x, H, W

class FocalSVTR(tf.keras.layers.Layer):
    def __init__(self,
                 img_size=[32, 128],
                 patch_size=[4, 4],
                 out_channels=256,                 
                 embed_dim=96,
                 depths=[3, 6, 3],
                 sub_k=[[2, 1], [2, 1], [1, 1]],
                 last_stage=False,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=layers.LayerNormalization,
                 patch_norm=True,               
                 focal_levels=[6, 6, 6],
                 focal_windows=[3, 3, 3],              
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 feat2d=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_layers = len(depths)
        self.embed_dim = [embed_dim * (2**i) for i in range(self.num_layers)]
        self.feat2d = feat2d
        self.patch_norm = patch_norm
        self.num_features = self.embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.last_stage = last_stage

        # Create patch embedding
        self.patch_embed = tf.keras.Sequential([
            ConvBNLayer(out_channels=self.embed_dim[0] // 2,
                        kernel_size=3,
                        stride=2,
                        padding='same',
                        name='fsvtr_patch_embed_cbn1'),
            ConvBNLayer(out_channels=self.embed_dim[0],
                        kernel_size=3,
                        stride=2,
                        padding='same',
                        name='fsvtr_patch_embed_cbn2'),
        ])

        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.patches_resolution = patches_resolution
        self.pos_drop = layers.Dropout(drop_rate)

        # stochastic depth decay rule
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers_list = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=self.embed_dim[i_layer],
                               out_dim=self.embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               input_resolution=patches_resolution,
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               downsample_kernel=sub_k[i_layer],
                               focal_level=focal_levels[i_layer],
                               focal_window=focal_windows[i_layer],
                               use_postln=use_postln,
                               use_postln_in_modulation=use_postln_in_modulation,
                               normalize_modulator=normalize_modulator)
            patches_resolution = [
                patches_resolution[0] // sub_k[i_layer][0],
                patches_resolution[1] // sub_k[i_layer][1]
            ]
            self.layers_list.append(layer)

        self.out_channels = self.num_features

        if last_stage:
            self.out_channels = out_channels
            self.last_conv = layers.Dense(self.out_channels, use_bias=False)
            self.hardswish = layers.Activation('hard_swish')
            self.dropout = layers.Dropout(0.1)

    def call(self, x, training=False):
        x = self.patch_embed(x, training=training)    # (b,h/4,w/4,embed_dim[0])

        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        # Flatten and transpose to (B, L, C)
        x = tf.reshape(x, [B, H * W, C])
        x = self.pos_drop(x, training=training)

        for layer in self.layers_list:
            x, H, W = layer(x, H, W, training=training)

        if self.feat2d:
            x = tf.reshape(x, [tf.shape(x)[0], H, W, self.out_channels])

        # Apply final processing for the last stage
        if self.last_stage:
            # Mean over H dimension
            x = tf.reduce_mean(tf.reshape(x, [tf.shape(x)[0], H, W, -1]), axis=1)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x, training=training)

        return x