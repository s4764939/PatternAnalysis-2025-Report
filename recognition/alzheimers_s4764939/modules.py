import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth

class LayerNorm2d(nn.LayerNorm):
    """ Layer Normalization for 2D inputs (N, C, H, W). """
    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

class ConvNeXtBlock(nn.Module):
    """
    A single block of the ConvNeXt architecture.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # Depthwise convolution
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # Pointwise/1x1 convs implemented as linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = StochasticDepth(p=drop_path, mode="row")

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    """
    A ConvNeXt model implementation built from scratch, following the original paper's
    design. This allows for customization of depths, dimensions, and other parameters.
    """
    def __init__(self, in_chans=1, num_classes=1, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # Initialize downsampling layers
        self.stages = nn.ModuleList() # Initialize stages

        # The stem and 3 intermediate downsampling layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # The four main stages of the network, each with multiple ConvNeXt blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # Final normalization layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # Global average pooling and normalization

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def create_convnext_model(num_classes=1, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.2):
    """
    Helper function to create a custom ConvNeXt model.

    Args:
        num_classes (int): Number of output classes.
        in_chans (int): Number of input channels (should be 1 for grayscale).
        depths (list): Number of blocks in each stage.
        dims (list): Channel dimensions in each stage.
        drop_path_rate (float): Stochastic depth rate (passed to the model).
    """
    model = ConvNeXt(in_chans=in_chans, num_classes=num_classes, depths=depths, dims=dims, drop_path_rate=drop_path_rate)
    return model

if __name__ == '__main__':
    # --- Model Creation and Forward Pass Test ---
    print("--- Testing Model Creation ---")
    
    # Create a model with parameters matching the 'small' variant.
    model = create_convnext_model(num_classes=1, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.2)
    
    print("Custom ConvNeXt-Small model created successfully from scratch.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Verify the input and output layers are correctly configured.
    print(f"Stem layer: {model.downsample_layers[0][0]}")
    print(f"Classifier layer: {model.head}")
    
    # Test a forward pass
    print("\n--- Testing Forward Pass ---")
    try:
        # Create a dummy input tensor to test the forward pass.
        dummy_input = torch.randn(4, 1, 224, 224) 
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (4, 1)
        print("Forward pass test successful.")
    except Exception as e:
        print(f"Error during forward pass: {e}")