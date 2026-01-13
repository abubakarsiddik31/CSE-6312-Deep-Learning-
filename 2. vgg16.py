"""
VGG-16 Implementation in PyTorch

Based on the paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
Authors: Karen Simonyan, Andrew Zisserman
Published in: International Conference on Learning Representations (ICLR) 2015

This implementation recreates the VGG-16 architecture, which achieved 7.3% top-5 error
on ImageNet classification, second place in ILSVRC 2014.

Key architectural features:
- 16 layers: 13 convolutional + 3 fully connected
- All convolutional layers use 3x3 filters with stride 1 and padding 1
- Max pooling with 2x2 filters and stride 2
- ReLU activations after every convolutional and fully-connected layer
- Dropout regularization (p=0.5) after first two FC layers
- Simple and uniform architecture design

Input: 224x224x3 RGB images
Output: 1000 class probabilities (ImageNet classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    """
    VGG-16 architecture implementation.

    The network consists of 16 layers: 13 convolutional layers organized in 5 blocks,
    followed by 3 fully-connected layers.
    Total parameters: approximately 138 million
    """

    def __init__(self, num_classes=1000):
        """
        Initialize VGG-16 architecture.

        Args:
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
        """
        super(VGG16, self).__init__()

        # Convolutional blocks
        # Block 1: 2 conv layers with 64 filters each, followed by maxpool
        # Input: 224x224x3 -> Output: 112x112x64 (after maxpool)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Block 2: 2 conv layers with 128 filters each, followed by maxpool
        # Input: 112x112x64 -> Output: 56x56x128 (after maxpool)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Block 3: 3 conv layers with 256 filters each, followed by maxpool
        # Input: 56x56x128 -> Output: 28x28x256 (after maxpool)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Block 4: 3 conv layers with 512 filters each, followed by maxpool
        # Input: 28x28x256 -> Output: 14x14x512 (after maxpool)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Block 5: 3 conv layers with 512 filters each, followed by maxpool
        # Input: 14x14x512 -> Output: 7x7x512 (after maxpool)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Max pooling layers (2x2, stride 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # FC6: 4096 neurons
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)

        # FC7: 4096 neurons
        self.fc7 = nn.Linear(4096, 4096)

        # FC8: num_classes neurons (1000 for ImageNet)
        self.fc8 = nn.Linear(4096, num_classes)

        # Dropout layers (p=0.5 as in the paper)
        self.dropout = nn.Dropout(p=0.5)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization
        as commonly used for VGG networks (though original paper used random initialization).
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # Xavier/Glorot initialization for conv layers
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for FC layers
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass through VGG-16.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Block 1: Conv1_1 -> ReLU -> Conv1_2 -> ReLU -> MaxPool
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)

        # Block 2: Conv2_1 -> ReLU -> Conv2_2 -> ReLU -> MaxPool
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)

        # Block 3: Conv3_1 -> ReLU -> Conv3_2 -> ReLU -> Conv3_3 -> ReLU -> MaxPool
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)

        # Block 4: Conv4_1 -> ReLU -> Conv4_2 -> ReLU -> Conv4_3 -> ReLU -> MaxPool
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)

        # Block 5: Conv5_1 -> ReLU -> Conv5_2 -> ReLU -> Conv5_3 -> ReLU -> MaxPool
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # FC6 -> ReLU -> Dropout
        x = F.relu(self.fc6(x))
        x = self.dropout(x)

        # FC7 -> ReLU -> Dropout
        x = F.relu(self.fc7(x))
        x = self.dropout(x)

        # FC8 (no activation - logits)
        x = self.fc8(x)
        return x


def create_vgg16(num_classes=1000, pretrained=False):
    """
    Create a VGG-16 model.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights (not implemented in this basic version)

    Returns:
        VGG16: Initialized VGG-16 model
    """
    model = VGG16(num_classes=num_classes)

    if pretrained:
        # Note: In a full implementation, you would load pretrained ImageNet weights here
        # For now, this is just a placeholder
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


# Alternative implementation using nn.Sequential for convolutional blocks
class VGG16Sequential(nn.Module):
    """
    Alternative VGG-16 implementation using nn.Sequential for cleaner code.
    This version is functionally identical but more compact.
    """

    def __init__(self, num_classes=1000):
        super(VGG16Sequential, self).__init__()

        # Define the feature extractor (convolutional layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Define the classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Forward pass through VGG-16."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Example usage and model summary
if __name__ == "__main__":
    # Create both versions of the model
    model1 = create_vgg16(num_classes=1000)
    model2 = VGG16Sequential(num_classes=1000)

    # Test with dummy input
    x = torch.randn(1, 3, 224, 224)
    output1 = model1(x)
    output2 = model2(x)

    print(f"Input shape: {x.shape}")
    print(f"VGG16 Output shape: {output1.shape}")
    print(f"VGG16Sequential Output shape: {output2.shape}")

    # Count parameters for both models
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())

    print(f"VGG16 parameters: {params1:,}")
    print(f"VGG16Sequential parameters: {params2:,}")

    # Verify outputs are identical
    print(f"Outputs identical: {torch.allclose(output1, output2)}")

    # Print model architecture (first version)
    print("\nVGG16 Architecture:")
    print(model1)

