# 이미지 캡셔닝 프로젝트

이미지를 입력받아 자연어 캡션을 자동으로 생성하는 딥러닝 모델입니다. ResNet 기반 Encoder와 Attention 메커니즘을 활용한 LSTM Decoder로 구성되어 있습니다.

## 📋 목차

- [프로젝트 소개](#프로젝트-소개)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [실행 방법](#실행-방법)
- [모델 구조](#모델-구조)
- [주요 기능](#주요-기능)
- [평가 지표](#평가-지표)
- [결과 시각화](#결과-시각화)

## 🎯 프로젝트 소개

이 프로젝트는 **Encoder-Decoder 아키텍처**를 기반으로 한 이미지 캡셔닝 모델을 구현합니다.

- **Encoder**: ResNet-18을 사용하여 이미지에서 특징을 추출
- **Decoder**: Bahdanau Attention 메커니즘을 활용한 다층 LSTM으로 캡션 생성
- **데이터셋**: Flickr8k 데이터셋 사용

### 주요 특징

- ✅ Attention 메커니즘으로 이미지의 특정 영역에 집중
- ✅ Attention heatmap 시각화 기능
- ✅ Label Smoothing을 통한 과적합 방지
- ✅ 반복 억제 메커니즘으로 자연스러운 캡션 생성
- ✅ BLEU, METEOR, ROUGE, CIDEr 등 다양한 평가 지표 지원

## 📁 프로젝트 구조

```
final_image_captioning/
├── main.py                      # 메인 학습 스크립트
├── requirements.txt             # 의존성 패키지 목록
├── readme.md                    # 프로젝트 문서
│
├── models/                      # 모델 정의
│   ├── __init__.py
│   └── image_caption_model.py   # Encoder-Decoder 통합 모델
│
├── modules/                     # 모듈 정의
│   ├── __init__.py
│   ├── resnet_18.py            # ResNet-18 Encoder
│   ├── encoder.py              # 인코더 유틸리티
│   ├── decoder.py              # LSTM Decoder with Attention
│   ├── preprocess.py           # 전처리 유틸리티
│   ├── evaluation.py           # 평가 지표 계산
│   └── attention_viz.py        # Attention 시각화
│
├── datasets/                    # 데이터셋 관련
│   ├── __init__.py
│   ├── flickr8k.py             # Flickr8k 데이터셋 로더
│   └── data/                    # 데이터 파일
│       ├── Flickr8k_images/    # 원본 이미지
│       ├── captions_preprocessed/  # 전처리된 캡션
│       │   ├── captions_padded.csv
│       │   ├── word2idx.json
│       │   └── idx2word.json
│       └── Flickr8k.token.txt   # 원본 캡션 파일
│
└── checkpoints/                 # 학습된 모델 저장
    ├── best_model.pth          # 최고 성능 모델
    ├── checkpoint_epoch_*.pth  # 에폭별 체크포인트
    └── attn_out/               # Attention heatmap 저장
        └── epoch_*/            # 에폭별 시각화 결과
```

## 🔧 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd final_image_captioning
```

### 2. 가상 환경 생성 및 활성화 (권장)

```bash
# Python 3.8 이상 필요
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 주요 의존성

- `torch>=2.0.0`: PyTorch 딥러닝 프레임워크
- `torchvision>=0.15.0`: 이미지 전처리 및 데이터셋
- `numpy>=1.24.0`, `pandas>=2.0.0`: 데이터 처리
- `nltk>=3.8.0`: 자연어 처리 및 평가 지표
- `Pillow>=10.0.0`: 이미지 처리
- `tqdm>=4.65.0`: 진행 상황 표시

## 📊 데이터 준비

### Flickr8k 데이터셋 다운로드

1. [Kaggle Flickr8k 데이터셋](https://www.kaggle.com/datasets/adityajn105/flickr8k)에서 데이터 다운로드
2. 프로젝트 루트에 압축 해제

### 데이터 전처리

데이터셋 클래스를 사용하여 자동으로 전처리됩니다:

```python
from datasets.flickr8k import Flickr8kDataset

dataset = Flickr8kDataset()
dataset.load_captions_to_df()  # 캡션 전처리 및 단어장 생성
```

전처리 결과:
- `datasets/data/captions_preprocessed/captions_padded.csv`: 패딩된 캡션
- `datasets/data/captions_preprocessed/word2idx.json`: 단어-인덱스 매핑
- `datasets/data/captions_preprocessed/idx2word.json`: 인덱스-단어 매핑

### 데이터 분할

- **학습 데이터**: 80%
- **검증 데이터**: 10%
- **테스트 데이터**: 10%

## 🚀 실행 방법

### 기본 학습 실행

```bash
python main.py
```

### 빠른 테스트 모드

`main.py`에서 `FAST_TEST = True`로 설정하면 빠른 테스트 모드로 실행됩니다:

```python
FAST_TEST = True  # 빠른 테스트 모드
```

빠른 테스트 모드 설정:
- 학습 데이터: 1000개 샘플
- 검증 데이터: 200개 샘플
- 에폭 수: 3
- 배치 크기: 128

### 실제 학습 모드 (기본값)

```python
FAST_TEST = False  # 실제 학습 모드
```

실제 학습 모드 설정:
- 전체 데이터셋 사용
- 에폭 수: 25
- 배치 크기: 128
- Hidden size: 512
- Embed size: 256

## 🏗️ 모델 구조

### 전체 아키텍처

```
ImageCaptionModel
├── Encoder (ResNet-18)
│   ├── Stem (Conv + BN + ReLU + MaxPool)
│   ├── Layer1-4 (BasicBlock × 2)
│   └── Global & Spatial Features
│       ├── Global Feature: [B, embed_size]
│       └── Spatial Feature: [B, H×W, embed_size]
│
└── Decoder (CaptionDecoder)
    ├── Embedding Layer
    ├── Bahdanau Attention
    ├── Multi-layer LSTM (2 layers)
    └── Linear Output Layer
```

### Encoder (ResNet-18)

- **입력**: 이미지 `[B, 3, 224, 224]`
- **출력**: 
  - Global feature: `[B, embed_size]` - 전체 이미지 특징
  - Spatial feature: `[B, H×W, embed_size]` - 공간적 특징 (Attention용)
  - Spatial size: `(H, W)` - 특징 맵 크기

### Decoder (LSTM with Attention)

- **Bahdanau Attention**: 이미지의 특정 영역에 집중
- **Multi-layer LSTM**: 2층 LSTM으로 문맥 이해
- **Teacher Forcing**: 학습 시 정답 캡션을 입력으로 사용

### 학습 전략

1. **Encoder 활성화**: Encoder 파라미터도 함께 학습
2. **Label Smoothing**: 과적합 방지 (smoothing=0.1)
3. **Gradient Clipping**: 그래디언트 폭발 방지 (max_norm=5.0)
4. **반복 억제**: 같은 단어 연속 반복 방지

## ⚙️ 주요 기능

### 1. 학습 모니터링

- 실시간 손실 표시 (학습/검증)
- Top-k 예측 출력 (디버깅용)
- 단어 빈도 분석

### 2. Attention 시각화

각 에폭마다 검증 데이터의 샘플 이미지에 대해:
- 생성된 캡션 출력
- 각 단어별 Attention heatmap 저장
- 저장 위치: `checkpoints/attn_out/epoch_{번호}/`

### 3. 체크포인트 저장

- **최고 성능 모델**: `checkpoints/best_model.pth`
- **에폭별 체크포인트**: `checkpoints/checkpoint_epoch_{번호}.pth`

### 4. 캡션 생성 옵션

- **Greedy Search**: 가장 높은 확률의 단어 선택
- **Sampling**: 확률 분포에서 샘플링
- **Top-k Sampling**: 상위 k개 후보 중 선택
- **반복 억제**: N-gram 반복 방지

## 📈 평가 지표

테스트 데이터셋에 대해 다음 지표를 계산합니다:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram 정확도
- **METEOR**: 동의어 및 형태소 매칭 고려
- **ROUGE-L**: 최장 공통 부분 수열 기반
- **CIDEr**: 이미지 캡셔닝 특화 지표

학습 완료 후 자동으로 평가가 실행되며 결과가 출력됩니다.

## 🎨 결과 시각화

### Attention Heatmap

각 단어 생성 시 모델이 주목한 이미지 영역을 시각화합니다:

```
checkpoints/attn_out/epoch_1/
├── 1000268201_693b08cb0e_0_a.png
├── 1000268201_693b08cb0e_1_dog.png
├── 1000268201_693b08cb0e_2_runs.png
└── ...
```

각 파일은 원본 이미지에 Attention 가중치를 오버레이한 결과입니다.

### 학습 로그 예시

```
Epoch 1/25
  학습 손실: 3.2456
  검증 손실: 2.9876
  소요 시간: 125.34초

[Epoch 1] Greedy 생성 결과 (attention heatmap):
이미지: 1000268201_693b08cb0e.jpg
  생성된 캡션: a dog runs in the grass
  ✓ 정상: 다양한 단어가 생성되었습니다
```

## 🔍 하이퍼파라미터

### 기본 설정

```python
# 모델 하이퍼파라미터
batch_size = 128
num_epochs = 25
hidden_size = 512
embed_size = 256
num_layers = 2
dropout = 0.1

# 학습 하이퍼파라미터
learning_rate = 0.001
weight_decay = 0.0001
gradient_clip = 5.0
label_smoothing = 0.1

# 캡션 생성 하이퍼파라미터
max_length = 20
repetition_penalty = 1.5
no_repeat_ngram_size = 3
```

## 📝 주요 파일 설명

### `main.py`

메인 학습 스크립트:
- 데이터 로드 및 전처리
- 모델 초기화 및 학습
- 검증 및 평가
- 체크포인트 저장
- Attention 시각화

### `models/image_caption_model.py`

통합 모델 클래스:
- Encoder와 Decoder 연결
- Forward pass (학습)
- `generate_caption()` (추론)

### `modules/decoder.py`

디코더 모듈:
- `BahdanauAttention`: Attention 메커니즘
- `CaptionDecoder`: LSTM 기반 디코더
- `step()`: 단계별 디코딩

### `modules/resnet_18.py`

ResNet-18 인코더:
- Global feature 추출
- Spatial feature 추출 (Attention용)

### `datasets/flickr8k.py`

데이터셋 로더:
- `Flickr8kImageCaptionDataset`: 이미지-캡션 쌍 데이터셋
- 자동 train/val/test 분할

## 🐛 문제 해결

### CUDA 메모리 부족

배치 크기를 줄이세요:
```python
batch_size = 64  # 또는 32
```

### 학습이 너무 느림

- `num_workers` 조정
- `FAST_TEST = True`로 빠른 테스트 모드 사용
- GPU 사용 확인: `torch.cuda.is_available()`

### 캡션이 반복됨

- `repetition_penalty` 증가 (예: 1.5 → 2.0)
- `no_repeat_ngram_size` 증가 (예: 3 → 4)

## 📚 참고 자료

- [Flickr8k 데이터셋](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [MS COCO 데이터셋](https://cocodataset.org/) (추가 데이터 필요 시)

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 👥 기여

이슈 및 풀 리퀘스트를 환영합니다!
