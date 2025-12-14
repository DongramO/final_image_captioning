"""
평가 지표 계산 모듈
"""

from typing import List
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
import numpy as np

# NLTK 데이터 다운로드 (최초 1회만)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


def calculate_bleu(references: List[List[str]], candidates: List[List[str]], n=4):
    """
    BLEU 점수 계산
    
    Args:
        references: 참조 캡션 리스트 (각 이미지당 여러 개의 참조 캡션)
                   예: [[["a", "dog", "runs"], ["dog", "running"]], ...]
        candidates: 생성된 캡션 리스트
                   예: [["a", "dog", "runs"], ...]
        n: n-gram 최대 차수 (기본값: 4)
        
    Returns:
        bleu_scores: BLEU 점수 딕셔너리 (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    """
    smoothing = SmoothingFunction().method1
    
    # 각 n-gram에 대한 BLEU 점수 계산
    bleu_scores = {}
    
    for i in range(1, n + 1):
        scores = []
        for refs, cand in zip(references, candidates):
            # sentence_bleu는 단일 참조용, 여러 참조가 있으면 첫 번째 사용
            # 더 정확한 평가를 위해 corpus_bleu 사용
            if len(refs) == 1:
                score = sentence_bleu(refs, cand, smoothing_function=smoothing, weights=tuple([1.0/i] * i))
            else:
                # 여러 참조가 있는 경우 corpus_bleu 스타일로 계산
                score = sentence_bleu(refs, cand, smoothing_function=smoothing, weights=tuple([1.0/i] * i))
            scores.append(score)
        
        bleu_scores[f'BLEU-{i}'] = np.mean(scores)
    
    # 전체 BLEU-4 점수 (가중 평균)
    weights = (0.25, 0.25, 0.25, 0.25)
    corpus_bleu_score = corpus_bleu(references, candidates, weights=weights, smoothing_function=smoothing)
    bleu_scores['BLEU-4'] = corpus_bleu_score
    
    return bleu_scores


def calculate_meteor(references: List[List[str]], candidates: List[List[str]]):
    """
    METEOR 점수 계산
    
    Args:
        references: 참조 캡션 리스트 (각 이미지당 여러 개의 참조 캡션)
        candidates: 생성된 캡션 리스트
        
    Returns:
        meteor_score: 평균 METEOR 점수 (0~1)
    """
    scores = []
    
    for refs, cand in zip(references, candidates):
        # METEOR는 단일 참조와 비교하므로, 여러 참조 중 최고 점수 사용
        best_score = 0.0
        for ref in refs:
            try:
                score = meteor_score([ref], cand)
                best_score = max(best_score, score)
            except:
                # METEOR 계산 실패 시 0점
                pass
        scores.append(best_score)
    
    return np.mean(scores) if scores else 0.0


def calculate_rouge(references: List[List[str]], candidates: List[List[str]]):
    """
    ROUGE 점수 계산 (간단한 구현)
    
    Args:
        references: 참조 캡션 리스트
        candidates: 생성된 캡션 리스트
        
    Returns:
        rouge_scores: ROUGE 점수 딕셔너리 (ROUGE-L)
    """
    def lcs_length(x, y):
        """최장 공통 부분 수열 길이 계산"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    rouge_l_scores = []
    
    for refs, cand in zip(references, candidates):
        best_score = 0.0
        for ref in refs:
            lcs = lcs_length(ref, cand)
            if len(ref) == 0 or len(cand) == 0:
                score = 0.0
            else:
                precision = lcs / len(cand) if len(cand) > 0 else 0.0
                recall = lcs / len(ref) if len(ref) > 0 else 0.0
                if precision + recall == 0:
                    score = 0.0
                else:
                    score = 2 * precision * recall / (precision + recall)
            best_score = max(best_score, score)
        rouge_l_scores.append(best_score)
    
    return {
        'ROUGE-L': np.mean(rouge_l_scores) if rouge_l_scores else 0.0
    }


def calculate_cider(references: List[List[str]], candidates: List[List[str]]):
    """
    CIDEr 점수 계산 (간단한 구현)
    
    Args:
        references: 참조 캡션 리스트
        candidates: 생성된 캡션 리스트
        
    Returns:
        cider_score: CIDEr 점수
    """
    def compute_tf_idf(sentences, n=4):
        """TF-IDF 계산"""
        # 모든 n-gram 수집
        all_ngrams = []
        for sent in sentences:
            for i in range(len(sent) - n + 1):
                all_ngrams.append(tuple(sent[i:i+n]))
        
        # 문서 빈도 계산
        doc_freq = Counter()
        for sent in sentences:
            sent_ngrams = set()
            for i in range(len(sent) - n + 1):
                sent_ngrams.add(tuple(sent[i:i+n]))
            for ngram in sent_ngrams:
                doc_freq[ngram] += 1
        
        # TF-IDF 계산
        num_docs = len(sentences)
        tf_idf = {}
        for sent in sentences:
            sent_tf = Counter()
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i:i+n])
                sent_tf[ngram] += 1
            
            for ngram, tf in sent_tf.items():
                df = doc_freq[ngram]
                idf = np.log(num_docs / (df + 1e-10))
                tf_idf[ngram] = tf * idf
        
        return tf_idf
    
    # 간단한 CIDEr 구현 (정확한 구현은 pycocotools 사용 권장)
    scores = []
    
    for refs, cand in zip(references, candidates):
        # 참조 캡션들의 TF-IDF 계산
        all_refs = refs
        ref_tfidf = compute_tf_idf(all_refs)
        
        # 후보 캡션의 TF-IDF 계산
        cand_tfidf = compute_tf_idf([cand])
        
        # 코사인 유사도 계산
        common_ngrams = set(ref_tfidf.keys()) & set(cand_tfidf.keys())
        if len(common_ngrams) == 0:
            score = 0.0
        else:
            dot_product = sum(ref_tfidf.get(ngram, 0) * cand_tfidf.get(ngram, 0) for ngram in common_ngrams)
            ref_norm = np.sqrt(sum(v**2 for v in ref_tfidf.values()))
            cand_norm = np.sqrt(sum(v**2 for v in cand_tfidf.values()))
            
            if ref_norm == 0 or cand_norm == 0:
                score = 0.0
            else:
                score = dot_product / (ref_norm * cand_norm)
        
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0