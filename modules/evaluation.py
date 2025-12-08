"""
평가 지표 계산 모듈
"""

from typing import List
import nltk


def calculate_bleu(references: List[List[str]], candidates: List[List[str]], n=4):
    """
    BLEU 점수 계산
    
    Args:
        references: 참조 캡션 리스트 (각 이미지당 여러 개의 참조 캡션)
        candidates: 생성된 캡션 리스트
        n: n-gram 최대 차수 (기본값: 4)
        
    Returns:
        bleu_score: BLEU 점수 (0~1)
    """
    # TODO: NLTK의 BLEU 점수 계산 함수 사용
    # from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    pass


def calculate_meteor(references: List[List[str]], candidates: List[List[str]]):
    """
    METEOR 점수 계산
    
    Args:
        references: 참조 캡션 리스트
        candidates: 생성된 캡션 리스트
        
    Returns:
        meteor_score: METEOR 점수 (0~1)
    """
    # TODO: NLTK의 METEOR 점수 계산 함수 사용
    # from nltk.translate.meteor_score import meteor_score
    pass


def calculate_rouge(references: List[List[str]], candidates: List[List[str]]):
    """
    ROUGE 점수 계산
    
    Args:
        references: 참조 캡션 리스트
        candidates: 생성된 캡션 리스트
        
    Returns:
        rouge_score: ROUGE 점수 딕셔너리
    """
    # TODO: ROUGE 점수 계산 (rouge-score 라이브러리 사용)
    pass


def calculate_cider(references: List[List[str]], candidates: List[List[str]]):
    """
    CIDEr 점수 계산
    
    Args:
        references: 참조 캡션 리스트
        candidates: 생성된 캡션 리스트
        
    Returns:
        cider_score: CIDEr 점수
    """
    # TODO: CIDEr 점수 계산 (pycocotools 사용)
    pass

