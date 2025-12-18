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
    
    # 빈 캡션 필터링
    valid_pairs = []
    for refs, cand in zip(references, candidates):
        # 빈 참조나 후보 캡션 제외
        if len(cand) == 0:
            continue
        # 빈 참조 캡션 필터링
        valid_refs = [ref for ref in refs if len(ref) > 0]
        if len(valid_refs) == 0:
            continue
        valid_pairs.append((valid_refs, cand))
    
    if len(valid_pairs) == 0:
        return {f'BLEU-{i}': 0.0 for i in range(1, n + 1)}
    
    # 각 n-gram에 대한 BLEU 점수 계산
    bleu_scores = {}
    
    for i in range(1, n + 1):
        scores = []
        for refs, cand in valid_pairs:
            # sentence_bleu는 여러 참조를 지원하므로 그대로 사용
            # weights는 i-gram까지만 사용 (예: i=2면 [0.5, 0.5])
            weights = tuple([1.0/i] * i)
            try:
                score = sentence_bleu(refs, cand, smoothing_function=smoothing, weights=weights)
                scores.append(score)
            except:
                # 계산 실패 시 0점
                scores.append(0.0)
        
        bleu_scores[f'BLEU-{i}'] = np.mean(scores) if scores else 0.0
    
    # BLEU-4는 이미 위에서 계산되었으므로 중복 제거
    # 대신 corpus_bleu로 전체 코퍼스 점수도 계산 (참고용)
    try:
        weights = (0.25, 0.25, 0.25, 0.25)
        valid_refs_list = [refs for refs, _ in valid_pairs]
        valid_cands_list = [cand for _, cand in valid_pairs]
        corpus_bleu_score = corpus_bleu(valid_refs_list, valid_cands_list, weights=weights, smoothing_function=smoothing)
        # BLEU-4는 개별 점수로 사용 (corpus_bleu는 참고용이므로 주석 처리)
        # bleu_scores['BLEU-4-corpus'] = corpus_bleu_score
    except:
        pass
    
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
        # 빈 캡션 처리
        if len(cand) == 0:
            scores.append(0.0)
            continue
        
        # 빈 참조 필터링
        valid_refs = [ref for ref in refs if len(ref) > 0]
        if len(valid_refs) == 0:
            scores.append(0.0)
            continue
        
        # METEOR는 단일 참조와 비교하므로, 여러 참조 중 최고 점수 사용
        best_score = 0.0
        for ref in valid_refs:
            try:
                score = meteor_score([ref], cand)
                if not np.isnan(score) and not np.isinf(score):
                    best_score = max(best_score, score)
            except (ValueError, ZeroDivisionError, AttributeError) as e:
                # 특정 예외만 처리 (모든 예외를 무시하지 않음)
                continue
            except Exception:
                # 기타 예외는 무시
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
        # 빈 캡션 처리
        if len(cand) == 0:
            rouge_l_scores.append(0.0)
            continue
        
        # 빈 참조 필터링
        valid_refs = [ref for ref in refs if len(ref) > 0]
        if len(valid_refs) == 0:
            rouge_l_scores.append(0.0)
            continue
        
        best_score = 0.0
        for ref in valid_refs:
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


def _prepare_pycocoevalcap_format(references: List[List[str]], candidates: List[List[str]]):
    """
    pycocoevalcap 형식으로 데이터 변환
    
    pycocoevalcap은 문자열 리스트를 기대합니다:
    - gts: {image_id: ["caption1", "caption2", ...]}
    - res: {image_id: ["caption"]}
    
    Returns:
        gts: {image_id: ["caption1", "caption2", ...]}
        res: {image_id: ["caption"]}
    """
    gts = {}
    res = {}
    
    for i, (refs, cand) in enumerate(zip(references, candidates)):
        # image_id는 문자열로 변환 (pycocoevalcap 요구사항)
        image_id = str(i)
        
        # 참조 캡션들 (단어 리스트를 문자열로 변환)
        gts[image_id] = [" ".join(ref) for ref in refs if len(ref) > 0]
        
        # 생성된 캡션 (단어 리스트를 문자열로 변환)
        if len(cand) > 0:
            res[image_id] = [" ".join(cand)]
        else:
            res[image_id] = [""]
    
    return gts, res


def calculate_cider(references: List[List[str]], candidates: List[List[str]]):
    """
    CIDEr 점수 계산 (표준 라이브러리: pycocoevalcap 사용)
    
    Args:
        references: 참조 캡션 리스트 (각 이미지당 여러 개의 참조 캡션)
        candidates: 생성된 캡션 리스트
        
    Returns:
        cider_score: CIDEr 점수 (0 이상)
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # pycocoevalcap 형식으로 변환
        gts, res = _prepare_pycocoevalcap_format(references, candidates)
        
        # CIDEr 계산기 생성 및 점수 계산
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        
        # score는 numpy array일 수 있으므로 float로 변환
        if isinstance(score, (list, np.ndarray)):
            score = float(score[0]) if len(score) > 0 else 0.0
        else:
            score = float(score)
        
        return max(0.0, score)  # 음수 방지
        
    except ImportError:
        # pycocoevalcap이 설치되지 않은 경우 fallback 사용
        print("⚠️ pycocoevalcap이 설치되지 않아 fallback CIDEr 계산을 사용합니다.")
        print("  표준 라이브러리를 사용하려면: pip install pycocoevalcap")
        return _calculate_cider_fallback(references, candidates)
    except Exception as e:
        # 기타 오류 발생 시 fallback 사용
        print(f"⚠️ pycocoevalcap CIDEr 계산 실패: {e}")
        print("  fallback CIDEr 계산을 사용합니다.")
        return _calculate_cider_fallback(references, candidates)


def _calculate_cider_fallback(references: List[List[str]], candidates: List[List[str]]):
    """
    CIDEr 점수 계산 (fallback 구현)
    
    Args:
        references: 참조 캡션 리스트
        candidates: 생성된 캡션 리스트
        
    Returns:
        cider_score: CIDEr 점수 (0 이상)
    """
    # 전체 코퍼스 수집 (모든 참조 캡션)
    all_corpus_sentences = []
    for refs in references:
        all_corpus_sentences.extend(refs)
    
    # 전체 코퍼스 크기
    corpus_size = len(all_corpus_sentences)
    if corpus_size == 0:
        return 0.0
    
    def compute_tf_idf_vector(sentence, corpus_doc_freq, corpus_size, n=4):
        """단일 문장에 대한 TF-IDF 벡터 계산"""
        # 문장이 n보다 짧으면 빈 벡터 반환
        if len(sentence) < n:
            return {}
        
        # 문장의 n-gram TF 계산
        sent_tf = Counter()
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i+n])
            sent_tf[ngram] += 1
        
        # TF-IDF 벡터 계산
        tf_idf = {}
        for ngram, tf in sent_tf.items():
            df = corpus_doc_freq.get(ngram, 0)
            # IDF 계산: log((N + 1) / (df + 1)) + 1 (음수 방지)
            idf = np.log((corpus_size + 1) / (df + 1)) + 1
            tf_idf[ngram] = tf * idf
        
        return tf_idf
    
    def compute_corpus_doc_freq(sentences, n=4):
        """전체 코퍼스에 대한 문서 빈도 계산"""
        doc_freq = Counter()
        for sent in sentences:
            # 문장이 n보다 짧으면 건너뛰기
            if len(sent) < n:
                continue
            sent_ngrams = set()
            for i in range(len(sent) - n + 1):
                sent_ngrams.add(tuple(sent[i:i+n]))
            for ngram in sent_ngrams:
                doc_freq[ngram] += 1
        return doc_freq
    
    # 1-4 gram에 대해 각각 계산하고 평균
    ngram_scores = []
    
    for n in range(1, 5):  # 1-gram to 4-gram
        # 전체 코퍼스에 대한 문서 빈도 계산
        corpus_doc_freq = compute_corpus_doc_freq(all_corpus_sentences, n=n)
        
        ngram_level_scores = []
        
        for refs, cand in zip(references, candidates):
            # 빈 캡션 처리
            if len(cand) == 0:
                ngram_level_scores.append(0.0)
                continue
            
            # 빈 참조 필터링
            valid_refs = [ref for ref in refs if len(ref) > 0]
            if len(valid_refs) == 0:
                ngram_level_scores.append(0.0)
                continue
            
            # 참조 캡션들의 평균 TF-IDF 벡터 계산
            ref_vectors = []
            for ref in valid_refs:
                ref_vec = compute_tf_idf_vector(ref, corpus_doc_freq, corpus_size, n=n)
                ref_vectors.append(ref_vec)
            
            # 참조 벡터들의 평균 계산
            all_ref_ngrams = set()
            for vec in ref_vectors:
                all_ref_ngrams.update(vec.keys())
            
            ref_avg_tfidf = {}
            for ngram in all_ref_ngrams:
                avg_val = np.mean([vec.get(ngram, 0) for vec in ref_vectors])
                if avg_val > 0:
                    ref_avg_tfidf[ngram] = avg_val
            
            # 후보 캡션의 TF-IDF 벡터 계산
            cand_tfidf = compute_tf_idf_vector(cand, corpus_doc_freq, corpus_size, n=n)
            
            # 코사인 유사도 계산
            common_ngrams = set(ref_avg_tfidf.keys()) & set(cand_tfidf.keys())
            if len(common_ngrams) == 0:
                score = 0.0
            else:
                dot_product = sum(ref_avg_tfidf.get(ngram, 0) * cand_tfidf.get(ngram, 0) for ngram in common_ngrams)
                ref_norm = np.sqrt(sum(v**2 for v in ref_avg_tfidf.values()))
                cand_norm = np.sqrt(sum(v**2 for v in cand_tfidf.values()))
                
                if ref_norm == 0 or cand_norm == 0:
                    score = 0.0
                else:
                    score = dot_product / (ref_norm * cand_norm)
                    # 코사인 유사도는 -1~1 범위이지만, TF-IDF는 모두 양수이므로 0~1 범위
                    score = max(0.0, score)  # 음수 방지
            
            ngram_level_scores.append(score)
        
        ngram_scores.append(np.mean(ngram_level_scores) if ngram_level_scores else 0.0)
    
    # 1-4 gram 점수의 평균
    return np.mean(ngram_scores) if ngram_scores else 0.0