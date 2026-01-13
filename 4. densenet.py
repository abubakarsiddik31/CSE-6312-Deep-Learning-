"""
DenseNet Implementation in PyTorch

Based on the paper: "Densely Connected Convolutional Networks"
Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Published in: Computer Vision and Pattern Recognition (CVPR) 2017

This implementation recreates the DenseNet architecture, which achieves state-of-the-art
performance with fewer parameters than ResNet through dense connectivity patterns.

Key architectural innovations:
- Dense connectivity: Each layer receives feature maps from all preceding layers
- Dense blocks: Groups of layers with dense connections
- Transition layers: Between dense blocks for downsampling
- Growth rate: Controls feature map growth per layer
- Bottleneck layers: 1x1 convolutions for parameter efficiency
- Feature reuse through concatenation instead of summation

Input: 224x224x3 RGB images
Output: 1000 class probabilities (ImageNet classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """
    Single dense layer used within DenseNet blocks.

    Each dense layer consists of: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv
    The output is concatenated with the input to form the dense connectivity pattern.
    """

    def __init__(self, in_channels, growth_rate):
        """
        Initialize a dense layer.

        Args:
            in_channels (int): Number of input channels (sum of all previous layers)
            growth_rate (int): Number of output feature maps (k in the paper)
        """
        super(DenseLayer, self).__init__()

        # Bottleneck 1x1 convolution
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, padding=0, bias=False)

        # Regular 3x3 convolution
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        Forward pass through dense layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor (growth_rate new feature maps)
        """
        # Bottleneck path
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        # Regular convolution
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out


class DenseBlock(nn.Module):
    """
    Dense block consisting of multiple dense layers with dense connectivity.
    """

    def __init__(self, num_layers, in_channels, growth_rate):
        """
        Initialize dense block.

        Args:
            num_layers (int): Number of dense layers in this block
            in_channels (int): Number of input channels
            growth_rate (int): Growth rate k (feature maps added per layer)
        """
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Each layer receives input from all previous layers
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass through dense block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Concatenated output from all dense layers
        """
        features = [x]

        for layer in self.layers:
            # Concatenate all previous features
            layer_input = torch.cat(features, dim=1)
            # Apply dense layer
            layer_output = layer(layer_input)
            # Add to feature list
            features.append(layer_output)

        # Concatenate all features
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.

    Performs downsampling and compression: BN -> 1x1 Conv -> AvgPool
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize transition layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels (after compression)
        """
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through transition layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Downsampled and compressed output
        """
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    """
    DenseNet architecture implementation.

    Supports different configurations based on number of layers per dense block.
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 compression_factor=0.5, num_classes=1000):
        """
        Initialize DenseNet.

        Args:
            growth_rate (int): Growth rate k (default: 32)
            block_config (tuple): Number of layers in each dense block (default: DenseNet-121 config)
            num_init_features (int): Number of feature maps in initial convolution (default: 64)
            compression_factor (float): Compression factor θ for transition layers (default: 0.5)
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
        """
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.compression_factor = compression_factor

        # Initial convolution
        # Input: 224x224x3 -> Output: 112x112x64
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)

        # Initial max pooling
        # Input: 112x112x64 -> Output: 56x56x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transition layers
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            # Add dense block
            dense_block = DenseBlock(num_layers, num_features, growth_rate)
            self.dense_blocks.append(dense_block)

            # Update number of features after dense block
            num_features = num_features + num_layers * growth_rate

            # Add transition layer (except after last dense block)
            if i < len(block_config) - 1:
                out_features = int(num_features * compression_factor)
                transition = TransitionLayer(num_features, out_features)
                self.transition_layers.append(transition)
                num_features = out_features

        # Final batch normalization
        self.bn_final = nn.BatchNorm2d(num_features)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(num_features, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass through DenseNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution + BN + ReLU + MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Dense blocks and transition layers
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)

        # Final BN + ReLU
        x = self.bn_final(x)
        x = self.relu(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_densenet121(num_classes=1000, pretrained=False):
    """
    Create DenseNet-121 model.

    DenseNet-121 configuration: [6, 12, 24, 16] layers per dense block

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights

    Returns:
        DenseNet: DenseNet-121 model
    """
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        compression_factor=0.5,
        num_classes=num_classes
    )

    if pretrained:
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


def create_densenet169(num_classes=1000, pretrained=False):
    """
    Create DenseNet-169 model.

    DenseNet-169 configuration: [6, 12, 32, 32] layers per dense block

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights

    Returns:
        DenseNet: DenseNet-169 model
    """
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        compression_factor=0.5,
        num_classes=num_classes
    )

    if pretrained:
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


def create_densenet201(num_classes=1000, pretrained=False):
    """
    Create DenseNet-201 model.

    DenseNet-201 configuration: [6, 12, 48, 32] layers per dense block

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights

    Returns:
        DenseNet: DenseNet-201 model
    """
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        num_init_features=64,
        compression_factor=0.5,
        num_classes=num_classes
    )

    if pretrained:
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


def create_densenet264(num_classes=1000, pretrained=False):
    """
    Create DenseNet-264 model.

    DenseNet-264 configuration: [6, 12, 64, 48] layers per dense block

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights

    Returns:
        DenseNet: DenseNet-264 model
    """
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        num_init_features=64,
        compression_factor=0.5,
        num_classes=num_classes
    )

    if pretrained:
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


# Example usage and model summary
if __name__ == "__main__":
    # Create DenseNet-121 (most common variant)
    model = create_densenet121(num_classes=1000)

    # Test with dummy input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count layers
    dense_layers = sum(len(block.layers) for block in model.dense_blocks)
    dense_blocks = len(model.dense_blocks)
    transition_layers = len(model.transition_layers)

    print(f"Dense layers: {dense_layers}")
    print(f"Dense blocks: {dense_blocks}")
    print(f"Transition layers: {transition_layers}")

    # Print architecture summary
    print("\nDenseNet-121 Architecture Summary:")
    print("=" * 50)
    print("Initial: Conv7x7 + BN + ReLU + MaxPool")
    print("Dense Block 1: 6 layers (growth rate 32)")
    print("Transition 1: 1x1 Conv + AvgPool")
    print("Dense Block 2: 12 layers (growth rate 32)")
    print("Transition 2: 1x1 Conv + AvgPool")
    print("Dense Block 3: 24 layers (growth rate 32)")
    print("Transition 3: 1x1 Conv + AvgPool")
    print("Dense Block 4: 16 layers (growth rate 32)")
    print("Final: BN + Global AvgPool + FC 1000")
    print("=" * 50)

    # Show different DenseNet variants
    print("\nDenseNet Variants:")
    variants = [
        ("DenseNet-121", (6, 12, 24, 16)),
        ("DenseNet-169", (6, 12, 32, 32)),
        ("DenseNet-201", (6, 12, 48, 32)),
        ("DenseNet-264", (6, 12, 64, 48))
    ]

    for name, config in variants:
        total_layers = sum(config) + 1  # +1 for initial conv
        print(f"{name}: {config} layers per block, {total_layers} total conv layers")
