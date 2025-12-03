import torch
import torch.nn as nn
from torchvision import models

class ResNet18_Grayscale(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # Weights setup
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.model = models.resnet18(weights=weights)
        
        # Modify Input Layer (3 channels -> 1 channel)
        original_weights = self.model.conv1.weight
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize with summed weights to preserve edge detection
        with torch.no_grad():
            self.model.conv1.weight[:] = torch.sum(original_weights, dim=1, keepdim=True)
            
        # Modify Output Layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)