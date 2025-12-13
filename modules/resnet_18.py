import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, embed_size=224, pretrained=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        
        self.bn1 = nn.BatchNorm2d(64),
        self.relu = nn.ReLU(inplace=True),
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)),
        self.fc = nn.Linear(512, embed_size)
        

    
    def forward(self, x):
        indentity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        out += identity
        return F.relu(out)