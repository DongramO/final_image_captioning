"""
이미지 처리 유틸리티
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ImageUtils:
    """
    이미지 전처리 및 변환 유틸리티 클래스
    """
    
    @staticmethod
    def get_transform(image_size=(224, 224), is_train=True):
        """
        이미지 전처리 변환 함수 생성
        
        Args:
            image_size: (height, width) 튜플
            is_train: 학습용 여부 (True면 데이터 증강 포함)
            
        Returns:
            transform: torchvision.transforms.Compose 객체
        """
        if is_train:
            # TODO: 학습용 변환 (리사이즈, 랜덤 크롭, 랜덤 플립 등)
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # TODO: 검증/테스트용 변환 (리사이즈, 정규화만)
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """
        이미지 파일 로드
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            image: PIL Image 객체
        """
        # TODO: 이미지 파일 로드 및 RGB 변환
        pass
    
    @staticmethod
    def resize_image(image: Image.Image, size: tuple) -> Image.Image:
        """
        이미지 리사이즈
        
        Args:
            image: PIL Image 객체
            size: (width, height) 튜플
            
        Returns:
            resized_image: 리사이즈된 PIL Image 객체
        """
        # TODO: 이미지 리사이즈
        pass
    
    @staticmethod
    def normalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
        """
        이미지 텐서 정규화
        
        Args:
            image_tensor: 이미지 텐서 [C, H, W]
            
        Returns:
            normalized_tensor: 정규화된 텐서
        """
        # TODO: ImageNet 평균/표준편차로 정규화
        pass

