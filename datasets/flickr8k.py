"""
Flickr8k 데이터셋 로더
"""

import os
from torch.utils.data import Dataset
from PIL import Image


class Flickr8kDataset(Dataset):
    """
    Flickr8k 데이터셋을 로드하는 클래스
    """
    
    def __init__(self, image_dir, captions_file, transform=None, vocab=None):
        """
        Args:
            image_dir: 이미지 파일들이 있는 디렉토리 경로
            captions_file: 캡션이 저장된 파일 경로
            transform: 이미지 전처리 변환 함수
            vocab: 단어장 객체
        """
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        # TODO: captions_file을 읽어서 self.captions와 self.image_ids 초기화
        
    def __len__(self):
        """데이터셋의 크기 반환"""
        # TODO: 데이터셋 크기 반환
        return 0
    
    def __getitem__(self, idx):
        """
        Args:
            idx: 데이터 인덱스
            
        Returns:
            image: 전처리된 이미지 텐서
            caption: 토큰화된 캡션 텐서
        """
        # TODO: 이미지 로드 및 전처리
        # TODO: 캡션 로드 및 토큰화
        pass

