"""
AlexNet Implementation in PyTorch

Based on the paper: "ImageNet Classification with Deep Convolutional Neural Networks"
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
Published in: Advances in Neural Information Processing Systems (NIPS) 2012

This implementation recreates the AlexNet architecture that won the ImageNet Large Scale
Visual Recognition Challenge (ILSVRC) in 2012, achieving a top-5 error rate of 15.3%.

Key architectural features:
- 5 convolutional layers with increasing depth
- 3 fully connected layers
- Local Response Normalization (LRN) after first two conv layers
- Max pooling after conv1, conv2, and conv5
- Dropout regularization before the last two FC layers
- ReLU activations throughout (except final output layer)

Input: 224x224x3 RGB images
Output: 1000 class probabilities (ImageNet classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet architecture implementation.

    The network consists of 8 layers: 5 convolutional layers followed by 3 fully-connected layers.
    Total parameters: approximately 60 million
    """

    def __init__(self, num_classes=1000):
        """
        Initialize AlexNet architecture.

        Args:
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
        """
        super(AlexNet, self).__init__()

        # Convolutional layers
        # Conv1: 96 filters, 11x11 kernel, stride 4, padding 2
        # Input: 224x224x3 -> Output: 55x55x96
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)

        # Conv2: 256 filters, 5x5 kernel, stride 1, padding 2
        # Input: 27x27x96 -> Output: 27x27x256 (after maxpool)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)

        # Conv3: 384 filters, 3x3 kernel, stride 1, padding 1
        # Input: 13x13x256 -> Output: 13x13x384
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        # Conv4: 384 filters, 3x3 kernel, stride 1, padding 1
        # Input: 13x13x384 -> Output: 13x13x384
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)

        # Conv5: 256 filters, 3x3 kernel, stride 1, padding 1
        # Input: 13x13x384 -> Output: 13x13x256
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        # FC6: 4096 neurons
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)

        # FC7: 4096 neurons
        self.fc7 = nn.Linear(4096, 4096)

        # FC8: num_classes neurons (1000 for ImageNet)
        self.fc8 = nn.Linear(4096, num_classes)

        # Dropout layers (p=0.5 as in the paper)
        self.dropout = nn.Dropout(p=0.5)

        # Initialize weights according to the paper's method
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using the method described in the paper.
        Uses Gaussian distribution with zero mean and specific variance for different layer types.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # Initialize conv weights with Gaussian distribution
                # Variance scales with fan-in (number of input connections)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # Initialize FC weights with Gaussian distribution
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass through AlexNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv1 + ReLU + LRN + MaxPool
        x = F.relu(self.conv1(x))
        x = self._local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # Conv2 + ReLU + LRN + MaxPool
        x = F.relu(self.conv2(x))
        x = self._local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # Conv3 + ReLU
        x = F.relu(self.conv3(x))

        # Conv4 + ReLU
        x = F.relu(self.conv4(x))

        # Conv5 + ReLU + MaxPool
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # FC6 + ReLU + Dropout
        x = F.relu(self.fc6(x))
        x = self.dropout(x)

        # FC7 + ReLU + Dropout
        x = F.relu(self.fc7(x))
        x = self.dropout(x)

        # FC8 (no activation - logits)
        x = self.fc8(x)
        return x

    def _local_response_norm(self, x, size=5, alpha=1e-4, beta=0.75, k=2):
        """
        Local Response Normalization as described in the AlexNet paper.

        This normalization helps with training deeper networks by creating competition
        for large activities among neuron outputs computed using different kernels.

        Args:
            x (torch.Tensor): Input tensor
            size (int): Local size (depth radius)
            alpha (float): Scaling parameter
            beta (float): Exponent parameter
            k (float): Offset parameter

        Returns:
            torch.Tensor: Normalized tensor
        """
        return F.local_response_norm(x, size=size, alpha=alpha, beta=beta, k=k)


def create_alexnet(num_classes=1000, pretrained=False):
    """
    Create an AlexNet model.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): If True, load pretrained weights (not implemented in this basic version)

    Returns:
        AlexNet: Initialized AlexNet model
    """
    model = AlexNet(num_classes=num_classes)

    if pretrained:
        # Note: In a full implementation, you would load pretrained ImageNet weights here
        # For now, this is just a placeholder
        print("Warning: Pretrained weights not available in this implementation")
        print("Model initialized with random weights")

    return model


# Example usage and model summary
if __name__ == "__main__":
    # Create model
    model = create_alexnet(num_classes=1000)

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

    # Print model architecture
    print("\nModel Architecture:")
    print(model)
