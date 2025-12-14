import torch
import torch.nn as nn
import torch.nn.functional as F

class Stem(nn.Module):
    def __init__(self, embed_size=224, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return x


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, down=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down = down

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        out += identity

        # Residual connection 본질인  F(x) = h(x) + x를 진행
        return self.relu(out)

class ResNet(nn.Module):
    """
    ResNet-18 (간단 구현)

    변경점:
    - embed_size가 주어졌을 때, attention을 위해 layer4의 feature map을 flatten하여
      spatial token [B, H*W, embed_size] 형태로도 반환할 수 있게 확장.
    - 반환 형태:
        * embed_size is None  -> x (기존: pooled -> proj)  [B, num_classes]
        * embed_size not None -> (global_feat, spatial_feat, (H, W))
            - global_feat  : [B, embed_size]
            - spatial_feat : [B, H*W, embed_size]
            - (H, W)       : layer4 feature map의 공간 크기
    """

    def __init__(self, num_classes=1000, embed_size=None, stride=1):
        super().__init__()

        self.stem = Stem()
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, block_size=2, stride=1)
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, block_size=2, stride=2)
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, block_size=2, stride=2)
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, block_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.embed_size = embed_size
        out_dim = embed_size if embed_size is not None else num_classes

        # global feature projection
        self.proj = nn.Linear(512 * BasicBlock.expansion, out_dim)

        # spatial feature projection (attention용)
        if embed_size is not None:
            self.proj_att = nn.Linear(512 * BasicBlock.expansion, embed_size)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, block_size, stride=1):

        down = None
        if stride != 1 or in_channels != out_channels:
            down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(in_channels, out_channels, stride=stride, down=down)]
        for _ in range(1, block_size):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, down=None))

        return nn.Sequential(*layers)

    def _init_weights(self):
        # ResNet 계열 기본: Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        if hasattr(self, "proj_att"):
            nn.init.normal_(self.proj_att.weight, std=0.02)
            nn.init.zeros_(self.proj_att.bias)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # [B, 512, H, W]

        # global
        pooled = self.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)
        global_feat = self.proj(pooled)  # [B, embed_size] 또는 [B, num_classes]

        if self.embed_size is None:
            # 기존 동작 유지
            return global_feat

        # spatial (attention 재료)
        B, C, H, W = feat_map.shape
        spatial = feat_map.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # [B, HW, 512]
        spatial = self.proj_att(spatial)  # [B, HW, embed_size]

        return global_feat, spatial, (H, W)
