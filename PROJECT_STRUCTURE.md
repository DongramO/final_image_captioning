# 이미지 캡셔닝 프로젝트 구조

## 📁 프로젝트 디렉토리 구조

```
final_image_captioning/
│
├── data/                      # 데이터 관련
│   ├── raw/                   # 원본 데이터 (Flickr8k, MS COCO 등)
│   ├── processed/             # 전처리된 데이터
│   └── vocab/                 # 단어장 파일
│
├── datasets/                  # 데이터셋 로더
│   ├── __init__.py
│   ├── flickr8k.py            # Flickr8k 데이터셋 로더
│   └── coco.py                # MS COCO 데이터셋 로더
│
├── modules/                   # 모델 모듈
│   ├── __init__.py
│   ├── encoder.py             # 이미지 인코더 (CNN 기반)
│   ├── decoder.py             # 캡션 디코더 (RNN/LSTM/Transformer)
│   ├── preprocess.py          # 전처리 함수들
│   └── evaluation.py          # 평가 지표 (BLEU, METEOR 등)
│
├── models/                    # 전체 모델 정의
│   ├── __init__.py
│   └── image_caption_model.py # Encoder-Decoder 통합 모델
│
├── utils/                     # 유틸리티 함수
│   ├── __init__.py
│   ├── tokenizer.py           # 토큰화 관련 함수
│   ├── vocabulary.py          # 단어장 생성 및 관리
│   ├── image_utils.py         # 이미지 처리 유틸리티
│   └── logger.py              # 로깅 유틸리티
│
├── config/                    # 설정 파일
│   ├── config.yaml            # 기본 설정 (하이퍼파라미터 등)
│   └── model_config.yaml      # 모델 아키텍처 설정
│
├── checkpoints/               # 모델 체크포인트 저장
│   └── .gitkeep
│
├── outputs/                   # 결과물 저장
│   ├── predictions/           # 생성된 캡션
│   ├── images/                # 결과 이미지
│   └── .gitkeep
│
├── logs/                      # 로그 파일
│   └── .gitkeep
│
├── notebooks/                 # Jupyter 노트북 (선택사항)
│   └── exploration.ipynb
│
├── main.py                    # 메인 실행 파일
├── train.py                   # 학습 스크립트
├── inference.py               # 추론 스크립트
├── evaluate.py                # 평가 스크립트
│
├── requirements.txt           # Python 패키지 의존성
├── .gitignore                 # Git 제외 파일
└── README.md                  # 프로젝트 설명서
```

## 📋 주요 파일 설명

### 1. 데이터 관련
- **data/raw/**: 원본 이미지와 캡션 파일 저장
- **data/processed/**: 전처리된 이미지와 토큰화된 텍스트 저장
- **data/vocab/**: 생성된 단어장 파일 저장

### 2. 모델 관련
- **modules/encoder.py**: CNN 기반 이미지 인코더 (ResNet, VGG 등)
- **modules/decoder.py**: RNN/LSTM/Transformer 기반 텍스트 디코더
- **models/image_caption_model.py**: Encoder-Decoder 통합 모델

### 3. 학습 및 추론
- **train.py**: 모델 학습 스크립트
- **inference.py**: 학습된 모델로 캡션 생성
- **evaluate.py**: 모델 성능 평가

### 4. 설정
- **config/config.yaml**: 하이퍼파라미터, 경로 등 설정
- **config/model_config.yaml**: 모델 아키텍처 설정

## 🔄 워크플로우

1. **데이터 준비**: `data/raw/`에 데이터 다운로드
2. **전처리**: `modules/preprocess.py`로 이미지/텍스트 전처리
3. **학습**: `train.py` 실행
4. **평가**: `evaluate.py`로 BLEU 점수 계산
5. **추론**: `inference.py`로 새로운 이미지에 대한 캡션 생성


전체 프로젝트 아키텍처 개요

1. 전체 데이터 흐름 (End-to-End Pipeline)
[원본 데이터] 
    ↓
[전처리 단계] → [데이터셋 로더] → [모델 학습] → [평가/추론]

2. 단계별 상세 흐름
Phase 1: 데이터 준비 및 전처리
data/raw/ (원본 이미지 + 캡션 텍스트)
    ↓
utils/tokenizer.py → 텍스트를 토큰으로 분리
    ↓
utils/vocabulary.py → 단어장 구축 (word2idx, idx2word)
    ↓
utils/image_utils.py → 이미지 리사이즈, 정규화
    ↓
data/processed/ (전처리된 데이터)
data/vocab/ (단어장 파일 저장)

주요 작업:
이미지: 224x224 리사이즈, ImageNet 정규화
텍스트: 토큰화 → 단어장 구축 → 인덱스 변환
특수 토큰: <pad>, <unk>, <sos>, <eos>

Phase 2: 데이터셋 로딩
datasets/flickr8k.py 또는 datasets/coco.py
    ↓
DataLoader (배치 생성)
    ↓
[이미지 텐서, 캡션 텐서, 길이 정보]

역할:
Flickr8kDataset / CocoDataset: PyTorch Dataset 상속
__getitem__(): 이미지와 캡션을 텐서로 반환
배치 단위로 데이터 제공

Phase 3: 모델 아키텍처 (Encoder-Decoder)
입력 이미지 [B, 3, 224, 224]
    ↓
┌─────────────────────────────────┐
│  ImageEncoder (modules/encoder.py) │
│  - ResNet50/VGG16 (CNN)         │
│  - 특징 추출                    │
└─────────────────────────────────┘
    ↓
이미지 특징 벡터 [B, embed_size]
    ↓
┌─────────────────────────────────┐
│  CaptionDecoder (modules/decoder.py) │
│  - LSTM/GRU                    │
│  - 시퀀스 생성                 │
└─────────────────────────────────┘
    ↓
출력 로짓 [B, seq_len, vocab_size]

모델 구조:
ImageEncoder: CNN으로 이미지 → 특징 벡터
CaptionDecoder: LSTM으로 특징 벡터 → 캡션 시퀀스
ImageCaptionModel: 두 모듈 통합

Phase 4: 학습 과정 (train.py - backward 포함)
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward Pass
        images, captions = batch
        outputs = model(images, captions)  # [B, seq_len, vocab_size]
        loss = criterion(outputs, captions)
        
        # 2. Backward Pass (역전파)
        optimizer.zero_grad()  # 그래디언트 초기화
        loss.backward()        # 그래디언트 계산 ⭐ 핵심!
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 3. 가중치 업데이트
        optimizer.step()

학습 설정:
손실 함수: CrossEntropyLoss (단어 예측)
옵티마이저: Adam
그래디언트 클리핑: 폭발 방지
체크포인트: checkpoints/에 저장

### Phase 5: 추론 과정 (inference.py)
새로운 이미지
    ↓
ImageEncoder → 이미지 특징
    ↓
CaptionDecoder.sample() → Greedy/Beam Search
    ↓
토큰 인덱스 시퀀스
    ↓
Vocabulary.decode() → 단어 리스트
    ↓
생성된 캡션 문자열

추론 특징 : 
model.eval(): 평가 모드
torch.no_grad(): 그래디언트 계산 비활성화
Beam Search 또는 Greedy Search로 캡션 생성

### Phase 6: 평가 (evaluate.py)
생성된 캡션 vs 참조 캡션
    ↓
modules/evaluation.py
    ↓
BLEU, METEOR, ROUGE, CIDEr 점수

3. 모듈 간 의존성 관계
main.py / train.py
    ├── models/image_caption_model.py
    │   ├── modules/encoder.py
    │   └── modules/decoder.py
    │
    ├── datasets/flickr8k.py 또는 coco.py
    │   ├── utils/tokenizer.py
    │   ├── utils/vocabulary.py
    │   └── utils/image_utils.py
    │
    ├── utils/logger.py
    │
    └── config/config.yaml

4. 파일별 역할 요약
파일/모듈	역할	주요 함수/클래스
train.py	학습 스크립트	학습 루프, backward 호출
inference.py	추론 스크립트	캡션 생성
evaluate.py	평가 스크립트	BLEU 등 점수 계산
models/image_caption_model.py	통합 모델	ImageCaptionModel
modules/encoder.py	이미지 인코더	ImageEncoder (CNN)
modules/decoder.py	캡션 디코더	CaptionDecoder (LSTM)
datasets/flickr8k.py	데이터셋 로더	Flickr8kDataset
utils/vocabulary.py	단어장 관리	Vocabulary
utils/tokenizer.py	텍스트 토큰화	Tokenizer
utils/image_utils.py	이미지 전처리	ImageUtils
config/config.yaml	설정 파일	하이퍼파라미터

5. 실행 흐름 예시
# 1. 데이터 준비vocab = Vocabulary()vocab.build_vocab(all_captions)  # 단어장 구축
# 2. 데이터셋 생성dataset = Flickr8kDataset(image_dir, captions_file, transform, vocab)dataloader = DataLoader(dataset, batch_size=32)
# 3. 모델 생성encoder = ImageEncoder('resnet50', embed_size=256)decoder = CaptionDecoder(embed_size=256, hidden_size=512, vocab_size=len(vocab))model = ImageCaptionModel(encoder, decoder, len(vocab))
# 4. 학습train(model, dataloader, ...)  # backward 포함
# 5. 추론caption = model.generate_caption(image, vocab)

6. 핵심 설계 원칙
모듈화: Encoder/Decoder 분리
재사용성: 데이터셋 로더를 여러 데이터셋에 적용
확장성: 설정 파일로 하이퍼파라미터 관리
추적성: 로깅 및 체크포인트 저장