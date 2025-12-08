"""
텍스트 토큰화 유틸리티
"""

import re
import nltk
from typing import List


class Tokenizer:
    """
    텍스트를 토큰으로 분리하는 클래스
    """
    
    def __init__(self, lowercase=True, remove_punctuation=False):
        """
        Args:
            lowercase: 소문자 변환 여부
            remove_punctuation: 구두점 제거 여부
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        # TODO: NLTK 데이터 다운로드 확인
        
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰 리스트로 변환
        
        Args:
            text: 입력 텍스트
            
        Returns:
            tokens: 토큰 리스트
        """
        # TODO: 텍스트 전처리 및 토큰화
        # - 소문자 변환
        # - 구두점 제거
        # - NLTK word_tokenize 사용
        pass
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        토큰 리스트를 텍스트로 변환
        
        Args:
            tokens: 토큰 리스트
            
        Returns:
            text: 복원된 텍스트
        """
        # TODO: 토큰을 공백으로 연결하여 텍스트로 변환
        pass

