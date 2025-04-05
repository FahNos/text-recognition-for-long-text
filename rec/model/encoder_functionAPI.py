import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

def focal_modulation(x, 
                     dim, 
                     focal_window, 
                     focal_level, 
                     max_kh=None, 
                     focal_factor=2, 
                     bias=True, 
                     proj_drop=0.0, 
                     use_postln_in_modulation=False,
                     normalize_modulator=False):
    """Functional implementation of Focal Modulation"""
    f = layers.Dense(2 * dim + (focal_level + 1), use_bias=bias)
    h = layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=bias)
    act = layers.Activation('gelu')
    proj = layers.Dense(dim)
    proj_drop_layer = layers.Dropout(proj_drop)
    
    # Apply initial dense projection
    x_proj = f(x)
    
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    
    # Split tensor
    q = x_proj[:, :, :, :C]
    ctx = x_proj[:, :, :, C:2*C]
    gates = x_proj[:, :, :, 2*C:]
    
    # Create focal layers
    focal_layers = []
    for k in range(focal_level):
        kernel_size = focal_factor * k + focal_window
        if max_kh is not None:
            k_h, k_w = [min(kernel_size, max_kh), kernel_size]
            kernel_size = (k_h, k_w)
        
        focal_layer = tf.keras.Sequential([
            layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                use_bias=False
            ),
            layers.Activation('gelu')
        ])
        focal_layers.append(focal_layer)
    
    # Context aggregation
    ctx_all = 0
    for l in range(focal_level):
        ctx_l = focal_layers[l](ctx)
        gate_l = gates[:, :, :, l:l+1]
        ctx_all = ctx_all + ctx_l * gate_l
    
    # Global context
    ctx_global = tf.reduce_mean(ctx, axis=[1, 2], keepdims=True)
    ctx_global = act(ctx_global)
    ctx_all = ctx_all + ctx_global * gates[:, :, :, focal_level:focal_level+1]
    
    # Normalize context
    if normalize_modulator:
        ctx_all = ctx_all / (focal_level + 1)
    
    # Focal modulation
    modulator = h(ctx_all)
    x_out = q * modulator
    
    # Optional layer normalization
    if use_postln_in_modulation:
        ln = layers.LayerNormalization(epsilon=1e-5)
        x_out = ln(x_out)
    
    # Post projection
    x_out = proj(x_out)
    x_out = proj_drop_layer(x_out)
    
    return x_out

def focal_net_block(x, 
                    H, 
                    W, 
                    dim, 
                    input_resolution=None, 
                    mlp_ratio=4.0, 
                    drop=0.0, 
                    drop_path=0.0, 
                    focal_level=1,
                    focal_window=3,
                    max_kh=None,
                    use_layerscale=False,
                    layerscale_value=1e-4,
                    use_postln=False,
                    use_postln_in_modulation=False,
                    normalize_modulator=False):
    """Functional implementation of FocalNetBlock"""
    norm1 = layers.LayerNormalization(epsilon=1e-5)
    norm2 = layers.LayerNormalization(epsilon=1e-5)
    
    # Dropout path
    def drop_path_func(inputs, training=False):
        if drop_path > 0:
            drop_path_layer = layers.Dropout(drop_path)
            return drop_path_layer(inputs, training=training)
        return inputs
    
    # MLPs
    mlp_hidden_dim = int(dim * mlp_ratio)
    mlp_layer = tf.keras.Sequential([
        layers.Dense(mlp_hidden_dim, activation='gelu'),
        layers.Dense(dim)
    ])
    
    # Shortcut
    shortcut = x
    
    # Layer scale
    gamma_1 = 1.0
    gamma_2 = 1.0
    if use_layerscale:
        gamma_1 = tf.Variable(tf.ones((dim,)) * layerscale_value, trainable=True)
        gamma_2 = tf.Variable(tf.ones((dim,)) * layerscale_value, trainable=True)
    
    # Modulation process
    x = tf.reshape(x, [-1, H, W, dim])
    x = focal_modulation(
        x, 
        dim=dim, 
        proj_drop=drop,
        focal_window=focal_window,
        focal_level=focal_level,
        max_kh=max_kh,
        use_postln_in_modulation=use_postln_in_modulation,
        normalize_modulator=normalize_modulator
    )
    x = tf.reshape(x, [-1, H * W, dim])
    
    # Normalization logic
    if not use_postln:
        x = norm1(x)
    
    # Residual connection and path dropout
    x = shortcut + drop_path_func(gamma_1 * x)
    
    # MLP processing
    if use_postln:
        x = x + drop_path_func(gamma_2 * mlp_layer(norm2(x)))
    else:
        x = x + drop_path_func(norm2(mlp_layer(x)))
    
    return x

def patch_embed(x, 
                img_size=(224, 224), 
                patch_size=[4, 4], 
                in_chans=3, 
                embed_dim=96, 
                use_conv_embed=False, 
                norm_layer=None, 
                is_stem=False):
    """Functional implementation of PatchEmbed"""
    patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    
    if use_conv_embed:
        if is_stem:
            kernel_size, padding, stride = 7, 'same', 4
        else:
            kernel_size, padding, stride = 3, 'same', 2
        
        conv_proj = layers.Conv2D(filters=embed_dim,
                                  kernel_size=kernel_size,
                                  strides=stride,
                                  padding=padding)
    else:
        conv_proj = layers.Conv2D(filters=embed_dim,
                                  kernel_size=patch_size,
                                  strides=patch_size,
                                  padding='valid')
    
    x = conv_proj(x)
    shape = tf.shape(x)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    
    x = tf.reshape(x, [B, H * W, C])
    
    if norm_layer is not None:
        norm = norm_layer(epsilon=1e-5)
        x = norm(x)
    
    return x, H, W

def basic_layer(x, 
                H, 
                W, 
                dim, 
                out_dim=None, 
                input_resolution=None, 
                depth=3, 
                mlp_ratio=4.0, 
                drop=0.0, 
                drop_path=0.0, 
                norm_layer=layers.LayerNormalization, 
                downsample=None, 
                downsample_kernel=[],
                use_checkpoint=False,
                focal_level=1,
                focal_window=1,
                use_conv_embed=False,
                use_layerscale=False,
                layerscale_value=1e-4,
                use_postln=False,
                use_postln_in_modulation=False,
                normalize_modulator=False):
    """Functional implementation of BasicLayer"""
    
    # Stochastic depth decay
    if isinstance(drop_path, (int, float)):
        drop_path = [drop_path] * depth
    
    # Apply blocks
    for i in range(depth):
        x = focal_net_block(
            x, 
            H, 
            W, 
            dim, 
            input_resolution=input_resolution,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path[i],
            focal_level=focal_level,
            focal_window=focal_window,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            use_postln=use_postln,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator
        )
    
    # Downsample if required
    if downsample is not None:
        x = tf.reshape(x, [tf.shape(x)[0], H, W, dim])
        x, H, W = patch_embed(
            x, 
            img_size=(H, W), 
            patch_size=downsample_kernel, 
            in_chans=dim, 
            embed_dim=out_dim, 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer
        )
    
    return x, H, W

def focal_svtr(
    x, 
    img_size=[32, 128],
    patch_size=[4, 4],
    out_channels=256,
    out_char_num=25,
    in_channels=3,
    embed_dim=96,
    depths=[3, 6, 3],
    sub_k=[[2, 1], [2, 1], [1, 1]],
    last_stage=False,
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=layers.LayerNormalization,
    patch_norm=True,
    use_checkpoint=False,
    focal_levels=[6, 6, 6],
    focal_windows=[3, 3, 3],
    use_conv_embed=False,
    use_layerscale=False,
    layerscale_value=1e-4,
    use_postln=False,
    use_postln_in_modulation=False,
    normalize_modulator=False,
    feat2d=False
):
    """Functional implementation of FocalSVTR"""
    
    # Patch embedding
    patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    
    # Initial patch embedding
    patch_embed_layers = tf.keras.Sequential([
        layers.Conv2D(embed_dim//2, kernel_size=3, strides=2, padding='same'),
        layers.Conv2D(embed_dim, kernel_size=3, strides=2, padding='same')
    ])
    
    x = patch_embed_layers(x)
    
    shape = tf.shape(x)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    
    x = tf.reshape(x, [B, H * W, C])
    x = layers.Dropout(drop_rate)(x)
    
    num_layers = len(depths)
    embed_dims = [embed_dim * (2**i) for i in range(num_layers)]
    
    # Stochastic depth
    dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
    
    for i_layer in range(num_layers):
        x, H, W = basic_layer(
            x, 
            H, 
            W, 
            dim=embed_dims[i_layer],
            out_dim=embed_dims[i_layer + 1] if i_layer < num_layers - 1 else None,
            input_resolution=patches_resolution,
            depth=depths[i_layer],
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            downsample_kernel=sub_k[i_layer],
            focal_level=focal_levels[i_layer],
            focal_window=focal_windows[i_layer],
            use_conv_embed=use_conv_embed
        )
        
        patches_resolution = [
            patches_resolution[0] // sub_k[i_layer][0],
            patches_resolution[1] // sub_k[i_layer][1]
        ]
    
    # 2D feature option
    if feat2d:
        x = tf.reshape(x, [tf.shape(x)[0], H, W, -1])
    
    # Last stage processing
    if last_stage:
        x = tf.reduce_mean(tf.reshape(x, [tf.shape(x)[0], H, W, -1]), axis=1)
        x = layers.Dense(out_channels, use_bias=False)(x)
        x = layers.Activation('hard_swish')(x)
        x = layers.Dropout(0.1)(x)
    
    return x