"""
이미지 인코더 모듈
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    CNN 기반 이미지 인코더
    ResNet, VGG 등의 사전 학습된 모델 사용
    """
    
    def __init__(self, encoder_type='resnet50', embed_size=256, pretrained=True):
        """
        Args:
            encoder_type: 인코더 타입 ('resnet50', 'resnet101', 'vgg16' 등)
            embed_size: 출력 임베딩 차원
            pretrained: 사전 학습된 가중치 사용 여부
        """
        super(ImageEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.embed_size = embed_size
        
        # TODO: encoder_type에 따라 적절한 모델 로드
        # 예: ResNet50, ResNet101, VGG16 등
        
        # TODO: 마지막 fully connected 레이어를 embed_size로 변경
        # self.linear = nn.Linear(feature_size, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        """
        Args:
            images: 입력 이미지 텐서 [batch_size, channels, height, width]
            
        Returns:
            features: 인코딩된 이미지 특징 [batch_size, embed_size]
        """
        # TODO: CNN을 통한 특징 추출
        # TODO: Linear 레이어를 통한 임베딩 변환
        pass

