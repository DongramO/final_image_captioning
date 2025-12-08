"""
이미지 캡셔닝 통합 모델
"""

import torch
import torch.nn as nn
from modules.encoder import ImageEncoder
from modules.decoder import CaptionDecoder


class ImageCaptionModel(nn.Module):
    """
    Encoder-Decoder 구조의 이미지 캡셔닝 모델
    """
    
    def __init__(self, encoder, decoder, vocab_size, embed_size=256, hidden_size=512):
        """
        Args:
            encoder: 이미지 인코더 모델
            decoder: 캡션 디코더 모델
            vocab_size: 단어장 크기
            embed_size: 임베딩 차원
            hidden_size: 히든 상태 차원
        """
        super(ImageCaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # TODO: 필요한 레이어 초기화 (예: 임베딩 레이어 등)
        
    def forward(self, images, captions=None, max_length=50):
        """
        Args:
            images: 입력 이미지 텐서 [batch_size, channels, height, width]
            captions: 캡션 텐서 [batch_size, seq_length] (학습 시)
            max_length: 최대 생성 길이 (추론 시)
            
        Returns:
            outputs: 생성된 캡션 로짓 [batch_size, seq_length, vocab_size]
        """
        # TODO: 이미지 인코딩
        # TODO: 캡션 디코딩
        pass
    
    def generate_caption(self, image, vocab, max_length=50, beam_size=5):
        """
        이미지로부터 캡션 생성 (추론)
        
        Args:
            image: 입력 이미지 텐서
            vocab: 단어장 객체
            max_length: 최대 생성 길이
            beam_size: Beam search 크기
            
        Returns:
            caption: 생성된 캡션 문자열
        """
        # TODO: Beam search 또는 Greedy search로 캡션 생성
        pass

