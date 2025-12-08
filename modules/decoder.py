"""
캡션 디코더 모듈
"""

import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    """
    RNN/LSTM 기반 캡션 디코더
    """
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        """
        Args:
            embed_size: 임베딩 차원 (단어 임베딩 + 이미지 특징 차원)
            hidden_size: LSTM 히든 상태 차원
            vocab_size: 단어장 크기
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
        """
        super(CaptionDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # TODO: 단어 임베딩 레이어
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # TODO: LSTM 레이어
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
        #                     batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # TODO: 출력 레이어
        # self.linear = nn.Linear(hidden_size, vocab_size)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, captions, lengths=None):
        """
        Args:
            features: 인코딩된 이미지 특징 [batch_size, embed_size]
            captions: 캡션 텐서 [batch_size, seq_length]
            lengths: 각 캡션의 실제 길이 (packed sequence용)
            
        Returns:
            outputs: 캡션 로짓 [batch_size, seq_length, vocab_size]
        """
        # TODO: 이미지 특징을 첫 번째 히든 상태로 사용
        # TODO: 단어 임베딩
        # TODO: LSTM을 통한 시퀀스 처리
        # TODO: 출력 레이어를 통한 로짓 생성
        pass
    
    def sample(self, features, max_length=50, start_token=None, end_token=None):
        """
        Greedy decoding으로 캡션 생성
        
        Args:
            features: 인코딩된 이미지 특징 [batch_size, embed_size]
            max_length: 최대 생성 길이
            start_token: 시작 토큰 인덱스
            end_token: 종료 토큰 인덱스
            
        Returns:
            sampled_ids: 생성된 토큰 인덱스 리스트
        """
        # TODO: Greedy search로 캡션 생성
        pass

