"""
전처리 함수 모듈
"""

import torch
from PIL import Image
from typing import List, Tuple


def preprocess_image(image_path: str, image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    이미지 전처리 함수
    
    Args:
        image_path: 이미지 파일 경로
        image_size: (height, width) 튜플
        
    Returns:
        image_tensor: 전처리된 이미지 텐서 [C, H, W]
    """
    # TODO: 이미지 로드
    # TODO: 리사이즈
    # TODO: 텐서 변환
    # TODO: 정규화
    pass


def preprocess_text(text: str, tokenizer=None, vocab=None) -> List[int]:
    """
    텍스트 전처리 및 토큰화 함수
    
    Args:
        text: 입력 텍스트
        tokenizer: 토크나이저 객체
        vocab: 단어장 객체
        
    Returns:
        token_indices: 토큰 인덱스 리스트
    """
    # TODO: 텍스트 토큰화
    # TODO: 단어장을 통한 인덱스 변환
    # TODO: 특수 토큰 추가 (SOS, EOS 등)
    pass

