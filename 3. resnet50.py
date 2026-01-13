"""
ResNet-50 Implementation in PyTorch

Based on the paper: "Deep Residual Learning for Image Recognition"
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Published in: Computer Vision and Pattern Recognition (CVPR) 2016

This implementation recreates the ResNet-50 architecture, which won the ILSVRC 2015
classification competition with a top-5 error rate of 3.57%.

Key architectural innovations:
- Residual connections (skip connections) to enable training of very deep networks
- Bottleneck residual blocks to reduce computational cost
- Identity mappings and projection shortcuts for skip connections
- Batch normalization after every convolutional layer
- Global average pooling before final classification layer

Input: 224x224x3 RGB images
Output: 1000 class probabilities (ImageNet classification)
"""

import torch
import torch.nn as nn


class BottleneckResidualBlock(nn.Module):
    """
    Bottleneck Residual Block used in ResNet-50 and deeper variants.

    This block uses 1x1 convolutions to reduce and then restore dimensions,
    making it more computationally efficient than the basic residual block.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initialize bottleneck residual block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the 3x3 convolution
            downsample (nn.Module): Downsampling layer for skip connection when dimensions change
        """
        super(BottleneckResidualBlock, self).__init__()

        # First 1x1 convolution: reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 convolution: main computation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Third 1x1 convolution: restore channels (expansion by factor of 4)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        # Skip connection (identity or projection)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through bottleneck residual block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        # Save input for skip connection
        identity = x

        # Main path: 1x1 -> BN -> ReLU -> 3x3 -> BN -> ReLU -> 1x1 -> BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual and apply ReLU
        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """
    ResNet-50 architecture implementation.

    The network consists of 50 layers: initial conv layer + 16 residual blocks + final FC layer.
    Total parameters: approximately 25.6 million
    """

    def __init__(self, num_classes=1000):
        """
        Initialize ResNet-50 architecture.

        Args:
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
        """
        super(ResNet50, self).__init__()

        # Initial convolutional layer
        # Input: 224x224x3 -> Output: 112x112x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Initial max pooling
        # Input: 112x112x64 -> Output: 56x56x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2: 3 residual blocks with 64 output channels (256 total due to expansion)
        # Input: 56x56x64 -> Output: 56x56x256
        self.layer2 = self._make_layer(64, 64, 3, stride=1)

        # Stage 3: 4 residual blocks with 128 output channels (512 total due to expansion)
        # Input: 56x56x256 -> Output: 28x28x512
        self.layer3 = self._make_layer(256, 128, 4, stride=2)

        # Stage 4: 6 residual blocks with 256 output channels (1024 total due to expansion)
        # Input: 28x28x512 -> Output: 14x14x1024
        self.layer4 = self._make_layer(512, 256, 6, stride=2)

        # Stage 5: 3 residual blocks with 512 output channels (2048 total due to expansion)
        # Input: 14x14x1024 -> Output: 7x7x2048
        self.layer5 = self._make_layer(1024, 512, 3, stride=2)

        # Global average pooling
        # Input: 7x7x2048 -> Output: 1x1x2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected layer
        # Input: 2048 -> Output: num_classes
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a ResNet layer consisting of multiple residual blocks.

        Args:
            in_channels (int): Number of input channels to the layer
            out_channels (int): Number of intermediate channels in bottleneck blocks
            num_blocks (int): Number of residual blocks in this layer
            stride (int): Stride for the first block (others have stride 1)

        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        layers = []

        # First block may need downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(BottleneckResidualBlock(in_channels, out_channels, stride, downsample))

        # Remaining blocks with stride 1
        for _ in range(1, num_blocks):
            layers.append(BottleneckResidualBlock(out_channels * 4, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization for conv layers
        and constant initialization for batch norm layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # Kaiming initialization for convolutional layers
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                # Constant initialization for batch normalization
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # Normal initialization for fully connected layers
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass through ResNet-50.

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

        # Residual layers
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten and final fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_resnet50(num_classes=1000, pretrained=False):
    """
    Create a ResNet-50 model.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights (not implemented in this basic version)

    Returns:
        ResNet50: Initialized ResNet-50 model
    """
    model = ResNet50(num_classes=num_classes)

    if pretrained:
        # Note: In a full implementation, you would load pretrained ImageNet weights here
        # For now, this is just a placeholder
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


# Example usage and model summary
if __name__ == "__main__":
    # Create model
    model = create_resnet50(num_classes=1000)

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
    conv_layers = sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))
    bn_layers = sum(1 for layer in model.modules() if isinstance(layer, nn.BatchNorm2d))
    residual_blocks = sum(1 for layer in model.modules() if isinstance(layer, BottleneckResidualBlock))

    print(f"Convolutional layers: {conv_layers}")
    print(f"Batch normalization layers: {bn_layers}")
    print(f"Residual blocks: {residual_blocks}")

    # Print model architecture summary
    print("\nResNet-50 Architecture Summary:")
    print("=" * 50)
    print("Stage 1: Conv7x7 + BN + ReLU + MaxPool")
    print("Stage 2: 3 residual blocks (64 channels)")
    print("Stage 3: 4 residual blocks (128 channels)")
    print("Stage 4: 6 residual blocks (256 channels)")
    print("Stage 5: 3 residual blocks (512 channels)")
    print("Final: Global AvgPool + FC 1000")
    print("=" * 50)
