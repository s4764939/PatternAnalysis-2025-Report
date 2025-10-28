import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
import math

# --- Helper: Truncated Normal Initialization ---
# (Included to avoid timm dependency)

def _trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.",)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _trunc_normal_(tensor, mean, std, a, b)

# --- Helper: Stochastic Depth (DropPath) ---
# (Included to avoid timm dependency)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (N, 1, 1, 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# --- Custom ConvNeXt Block ---
# Based on Figure 4 of the ConvNeXt paper

class ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt Block, built layer-by-layer.
    This implementation uses the "channels_last" memory format internally
    for its linear layers, just like the original paper's design.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise 7x7 convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        
        # LayerNorm (applied on channels_last)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise 1x1 conv (expansion)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        
        self.act = nn.GELU()
        
        # Pointwise 1x1 conv (projection)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # LayerScale
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            
        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        
        # 1. Depthwise Conv
        x = self.dwconv(x)
        
        # 2. Permute to channels_last (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1) 
        
        # 3. LayerNorm
        x = self.norm(x)
        
        # 4. Pointwise (Linear) layers
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 5. LayerScale
        if self.gamma is not None:
            x = self.gamma * x
            
        # 6. Permute back to channels_first (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2) 

        # 7. Add residual connection with DropPath
        x = input + self.drop_path(x)
        
        return x

# --- Custom LayerNorm for (N, C, H, W) data ---
# This applies LayerNorm *after* permuting, to match the paper's
# "normalization layer" after the stem and between stages.

class PermuteLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x

# --- The Custom ConvNeXt Model ---

class CustomConvNeXt(nn.Module):
    """
    A custom, layer-by-layer implementation of ConvNeXt.
    
    Args:
        in_chans (int): Number of input image channels.
        num_classes (int): Number of classes for classification head.
        depths (list(int)): Number of blocks at each stage.
        dims (list(int)): Feature dimension at each stage.
        drop_path_rate (float): Stochastic depth rate.
        layer_scale_init_value (float): Init value for Layer Scale.
    """
    def __init__(self, in_chans=1, num_classes=1, 
                 depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        
        # --- 1. Stem (Patchify) ---
        # 4x4 non-overlapping conv + LayerNorm
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            PermuteLayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # --- 2. Downsampling Layers ---
        # Add 3 downsampling layers (between stages 1-2, 2-3, 3-4)
        # 2x2 conv + LayerNorm
        for i in range(3):
            downsample_layer = nn.Sequential(
                PermuteLayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # --- 3. Stages ---
        # Add 4 stages of ConvNeXt Blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i], 
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

        # --- 4. Head ---
        # Final LayerNorm
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        # Classification Head
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # Init head linear layer
        trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Pass through stem + 3 downsamplers and 4 stages
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # Global Average Pooling (take mean over H, W)
        x_pooled = x.mean([-2, -1]) # (N, C, H, W) -> (N, C)
        
        # Final LayerNorm
        return self.norm(x_pooled)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# --- Model Creator Function ---
# This is a drop-in replacement for your original `create_convnext_model`

def create_convnext_model(num_classes=1, in_chans=1, 
                                 depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]):
    """
    Creates a *custom* ConvNeXt model, built layer-by-layer.
    
    Args:
        num_classes (int): Number of output classes.
        in_chans (int): Number of input channels.
        depths (list(int)): Number of blocks at each stage.
        dims (list(int)): Feature dimension at each stage.
    """
    # This matches the ConvNeXt-S architecture by default
    model = CustomConvNeXt(
        in_chans=in_chans, 
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        drop_path_rate=0.2
    )

    # Replicate the head customization from your original modules.py
    # This ensures the new model is a seamless replacement
    num_ftrs = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

if __name__ == '__main__':
    # Test creating the model
    # This creates the default architecture (matching ConvNeXt-S)
    model = create_custom_convnext_model(
        num_classes=1, 
        in_chans=1
    )
    
    print("Custom ConvNeXt model created successfully.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test a forward pass
    try:
        dummy_input = torch.randn(2, 1, 224, 224) # (B, C, H, W)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

