"""
단어장(Vocabulary) 관리 클래스
"""

from collections import Counter
from typing import List, Dict


class Vocabulary:
    """
    단어와 인덱스를 매핑하는 단어장 클래스
    """
    
    def __init__(self, min_freq=2):
        """
        Args:
            min_freq: 단어장에 포함될 최소 빈도수
        """
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
        # 특수 토큰 추가
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<sos>'  # Start of Sentence
        self.EOS_TOKEN = '<eos>'  # End of Sentence
        
    def build_vocab(self, captions: List[List[str]]):
        """
        캡션 리스트로부터 단어장 구축
        
        Args:
            captions: 토큰화된 캡션 리스트의 리스트
        """
        # TODO: 모든 단어의 빈도수 계산
        # TODO: min_freq 이상인 단어만 단어장에 추가
        # TODO: 특수 토큰 추가
        pass
    
    def add_word(self, word: str):
        """
        단어를 단어장에 추가
        
        Args:
            word: 추가할 단어
        """
        # TODO: 단어를 단어장에 추가하고 인덱스 할당
        pass
    
    def word_to_idx(self, word: str) -> int:
        """
        단어를 인덱스로 변환
        
        Args:
            word: 입력 단어
            
        Returns:
            idx: 단어의 인덱스 (없으면 UNK 토큰 인덱스)
        """
        # TODO: 단어를 인덱스로 변환
        pass
    
    def idx_to_word(self, idx: int) -> str:
        """
        인덱스를 단어로 변환
        
        Args:
            idx: 단어 인덱스
            
        Returns:
            word: 인덱스에 해당하는 단어
        """
        # TODO: 인덱스를 단어로 변환
        pass
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        토큰 리스트를 인덱스 리스트로 변환
        
        Args:
            tokens: 토큰 리스트
            
        Returns:
            indices: 인덱스 리스트
        """
        # TODO: 토큰 리스트를 인덱스 리스트로 변환
        pass
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        인덱스 리스트를 토큰 리스트로 변환
        
        Args:
            indices: 인덱스 리스트
            
        Returns:
            tokens: 토큰 리스트
        """
        # TODO: 인덱스 리스트를 토큰 리스트로 변환
        pass
    
    def __len__(self):
        """단어장 크기 반환"""
        return len(self.word2idx)

