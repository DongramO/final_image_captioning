# 이미지 캡셔닝 프로젝트 구조

## 📁 프로젝트 디렉토리 구조

```
final_image_captioning/
│
├── data/                      # 데이터 관련
│   ├── raw/                   # 원본 데이터 (Flickr8k)
│   ├── processed/             # 전처리된 데이터
│   └── vocab/                 # 단어장 파일
│
├── datasets/                  # 데이터셋 로더
│   ├── __init__.py
│   ├── flickr8k.py            # Flickr8k 데이터셋 로더 및 전처리
│   └── data/                  # 실제 데이터 파일
│       ├── Flickr8k_images/   # 원본 이미지 파일들
│       └── captions_preprocessed/  # 전처리된 캡션
│           ├── captions_padded.csv
│           ├── word2idx.json
│           └── idx2word.json
│
├── modules/                   # 모델 모듈
│   ├── __init__.py
│   ├── resnet_18.py           # ResNet-18 인코더 구현
│   ├── encoder.py             # 인코더 유틸리티 함수
│   ├── decoder.py             # LSTM 디코더 + Bahdanau Attention
│   ├── preprocess.py          # 전처리 함수들
│   ├── evaluation.py          # 평가 지표 (BLEU, METEOR, ROUGE, CIDEr)
│   └── attention_viz.py       # Attention heatmap 시각화
│
├── models/                    # 전체 모델 정의
│   ├── __init__.py
│   └── image_caption_model.py # Encoder-Decoder 통합 모델
│
├── checkpoints/               # 모델 체크포인트 저장
│   ├── best_model.pth        # 최고 성능 모델
│   ├── checkpoint_epoch_*.pth # 에폭별 체크포인트
│   └── attn_out/             # Attention heatmap 저장
│       └── epoch_*/          # 에폭별 시각화 결과
│
├── outputs/                   # 결과물 저장
│   ├── predictions/           # 생성된 캡션
│   └── images/                # 결과 이미지
│
├── logs/                      # 로그 파일
│
├── notebooks/                 # Jupyter 노트북 (선택사항)
│
├── main.py                    # 메인 학습 스크립트
├── visual.py                  # 시각화 스크립트
├── requirements.txt           # Python 패키지 의존성
├── readme.md                  # 프로젝트 설명서
└── PROJECT_STRUCTURE.md       # 프로젝트 구조 문서 (본 파일)
```

## 📋 주요 파일 상세 설명

### 1. 메인 실행 파일

#### `main.py` - 학습 스크립트
**역할**: 전체 학습 파이프라인 실행

**주요 함수**:
- `train_one_epoch()`: 한 에폭 학습 (forward, backward, optimizer.step 포함)
- `validate()`: 검증 데이터셋 평가
- `evaluate_model()`: 테스트 데이터셋 평가 (BLEU, METEOR, ROUGE, CIDEr)
- `save_checkpoint()` / `load_checkpoint()`: 모델 체크포인트 저장/로드
- `analyze_word_frequency()`: 단어 빈도 분석
- `get_topk_predictions()`: Top-k 예측 출력 (디버깅용)

**주요 클래스**:
- `LabelSmoothingCrossEntropy`: Label Smoothing 적용 손실 함수

**학습 설정**:
- `FAST_TEST = False`: 실제 학습 모드 (25 에폭, 전체 데이터)
- `FAST_TEST = True`: 빠른 테스트 모드 (3 에폭, 1000개 샘플)
- 배치 크기: 128
- 학습률: 0.001
- Label Smoothing: 0.1
- Gradient Clipping: 5.0

### 2. 모델 정의

#### `models/image_caption_model.py` - 통합 모델
**역할**: Encoder와 Decoder를 연결하는 통합 모델

**주요 클래스**:
- `ImageCaptionModel`: Encoder-Decoder 통합 모델

**주요 메서드**:
- `forward()`: 학습 시 forward pass (Teacher Forcing)
- `generate_caption()`: 추론 시 캡션 생성
  - Greedy search / Sampling / Top-k sampling 지원
  - 반복 억제 메커니즘 (repetition_penalty, no_repeat_ngram_size)
  - Attention 정보 반환 옵션 (`return_attention=True`)

**입력/출력**:
- 입력: 이미지 `[B, 3, 224, 224]`
- 출력: 로짓 `[B, seq_len, vocab_size]` (학습) 또는 캡션 문자열 리스트 (추론)

#### `modules/resnet_18.py` - ResNet-18 인코더
**역할**: 이미지에서 특징 추출

**주요 클래스**:
- `Stem`: 초기 컨볼루션 레이어
- `BasicBlock`: ResNet 기본 블록
- `ResNet`: ResNet-18 전체 구조

**출력 형태**:
- `embed_size`가 주어지면: `(global_feat, spatial_feat, (H, W))`
  - `global_feat`: `[B, embed_size]` - 전체 이미지 특징
  - `spatial_feat`: `[B, H*W, embed_size]` - 공간적 특징 (Attention용)
  - `(H, W)`: 특징 맵 크기 (예: (7, 7))

**구조**:
- Stem → Layer1 → Layer2 → Layer3 → Layer4
- Global feature: AdaptiveAvgPool2d + Linear projection
- Spatial feature: Layer4 출력을 flatten + Linear projection

#### `modules/decoder.py` - LSTM 디코더 + Attention
**역할**: 이미지 특징으로부터 캡션 생성

**주요 클래스**:
- `BahdanauAttention`: Additive Attention 메커니즘
  - 입력: `encoder_out [B, P, E]`, `decoder_h [B, H]`
  - 출력: `context [B, E]`, `alpha [B, P]`
  - Temperature scaling으로 attention을 더 날카롭게 만듦
- `CaptionDecoder`: 다층 LSTM 디코더
  - Embedding → Attention → LSTM (2 layers) → Linear
  - Context와 word embedding을 concat하여 LSTM 입력으로 사용
  - 학습 가능한 context 가중치 (`context_weight`)
- `LSTMCell`: 커스텀 LSTM 셀 구현

**주요 메서드**:
- `init_hidden_state()`: Global feature로 초기 hidden state 계산
- `step()`: 한 단계 디코딩 (추론용)
- `forward()`: 전체 시퀀스 디코딩 (학습용, Teacher Forcing)

**입력/출력**:
- 입력: `features [B, E]`, `captions [B, T]`, `encoder_out [B, P, E]` (optional)
- 출력: `outputs [B, T, vocab_size]`, `alphas [B, T, P]` (optional)

### 3. 데이터셋

#### `datasets/flickr8k.py` - Flickr8k 데이터셋 로더
**역할**: Flickr8k 데이터셋 로드 및 전처리

**주요 클래스**:
- `Flickr8kDataset`: 전처리용 클래스
  - `load_captions_to_df()`: 캡션 전처리 및 단어장 생성
  - `preprocess_image()`: 이미지 전처리
- `Flickr8kImageCaptionDataset`: PyTorch Dataset
  - `__getitem__()`: 이미지와 캡션 텐서 반환
  - 자동 train/val/test 분할 (80/10/10)
- `Flickr8kImageOnlyDataset`: 이미지만 로드하는 데이터셋

**전처리 과정**:
1. 텍스트 정제: 소문자 변환, 특수문자 제거
2. 토큰화: 공백 기준 분리
3. 단어장 구축: 특수 토큰 (`<pad>`, `<start>`, `<end>`, `<unk>`) + 단어들
4. 인덱스 변환: 단어 → 인덱스
5. 패딩: 최대 길이 20으로 패딩

**출력 파일**:
- `captions_padded.csv`: 패딩된 캡션
- `word2idx.json`: 단어 → 인덱스 매핑
- `idx2word.json`: 인덱스 → 단어 매핑

### 4. 평가 및 시각화

#### `modules/evaluation.py` - 평가 지표
**역할**: 다양한 평가 지표 계산

**주요 함수**:
- `calculate_bleu()`: BLEU-1, BLEU-2, BLEU-3, BLEU-4 점수
- `calculate_meteor()`: METEOR 점수 (동의어 고려)
- `calculate_rouge()`: ROUGE-L 점수
- `calculate_cider()`: CIDEr 점수 (이미지 캡셔닝 특화)

**사용 라이브러리**:
- NLTK: BLEU, METEOR 계산
- pycocotools: CIDEr 계산

#### `modules/attention_viz.py` - Attention 시각화
**역할**: Attention heatmap을 이미지에 오버레이하여 저장

**주요 함수**:
- `save_attention_overlays()`: 각 단어별 attention heatmap 저장
  - 입력: PIL Image, 단어 리스트, alpha 리스트, spatial 크기
  - 출력: 각 단어별 heatmap 이미지 파일

**저장 형식**:
- 파일명: `{prefix}_{step}_{word}.png`
- 예: `1000268201_693b08cb0e_0_a.png`, `1000268201_693b08cb0e_1_dog.png`

### 5. 유틸리티

#### `modules/encoder.py` - 인코더 유틸리티
**역할**: 인코더 관련 유틸리티 함수

**주요 함수**:
- `encode_images()`: 배치 이미지를 특징 벡터로 인코딩

#### `modules/preprocess.py` - 전처리 유틸리티
**역할**: 이미지/텍스트 전처리 함수

## 🔄 전체 데이터 흐름 (End-to-End Pipeline)

### Phase 1: 데이터 준비 및 전처리

```
원본 데이터 (Flickr8k)
    ↓
datasets/flickr8k.py::Flickr8kDataset.load_captions_to_df()
    ↓
텍스트 정제 (소문자, 특수문자 제거)
    ↓
토큰화 (공백 기준 분리)
    ↓
단어장 구축 (word2idx, idx2word)
    ↓
인덱스 변환 + 패딩 (최대 길이 20)
    ↓
저장: captions_padded.csv, word2idx.json, idx2word.json
```

**특수 토큰**:
- `<pad>` (idx=0): 패딩 토큰
- `<start>` (idx=1): 시작 토큰
- `<end>` (idx=2): 종료 토큰
- `<unk>` (idx=3): 알 수 없는 단어

### Phase 2: 데이터셋 로딩

```
datasets/flickr8k.py::Flickr8kImageCaptionDataset
    ↓
이미지 로드 (PIL Image)
    ↓
Transform 적용 (Resize 224x224, ToTensor, Normalize)
    ↓
캡션 텐서 변환 (LongTensor)
    ↓
DataLoader (배치 생성)
    ↓
출력: (images [B, 3, 224, 224], captions [B, T])
```

**데이터 분할**:
- 학습: 80%
- 검증: 10%
- 테스트: 10%

### Phase 3: 모델 아키텍처 (Encoder-Decoder with Attention)

```
입력 이미지 [B, 3, 224, 224]
    ↓
┌─────────────────────────────────────┐
│  ResNet-18 Encoder                  │
│  (modules/resnet_18.py)             │
│  - Stem → Layer1-4                  │
│  - Global feature: [B, embed_size]  │
│  - Spatial feature: [B, H*W, E]     │
└─────────────────────────────────────┘
    ↓
(global_feat, spatial_feat, (H, W))
    ↓
┌─────────────────────────────────────┐
│  CaptionDecoder                     │
│  (modules/decoder.py)               │
│  - Embedding                        │
│  - BahdanauAttention                │
│  - Multi-layer LSTM (2 layers)      │
│  - Linear Output                    │
└─────────────────────────────────────┘
    ↓
출력 로짓 [B, seq_len, vocab_size]
```

**Attention 메커니즘**:
1. Encoder의 spatial feature `[B, P, E]`와 Decoder의 hidden state `[B, H]`를 입력
2. Attention score 계산: `tanh(W_enc * encoder_out + W_dec * decoder_h)`
3. Softmax로 attention weight `alpha [B, P]` 계산
4. Weighted sum으로 context `[B, E]` 생성
5. Context와 word embedding을 concat하여 LSTM 입력으로 사용

### Phase 4: 학습 과정 (main.py)

```python
for epoch in range(num_epochs):
    # 1. 학습
    for batch in train_loader:
        images, captions = batch  # [B, 3, 224, 224], [B, T]
        
        # Forward Pass
        outputs = model(images, captions)  # [B, T, vocab_size]
        targets = captions[:, 1:]  # Teacher Forcing: <start> 제외
        
        # Loss 계산 (Label Smoothing 적용)
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    
    # 2. 검증
    val_loss = validate(model, val_loader, ...)
    
    # 3. Attention 시각화 (매 에폭마다)
    sample_image = val_dataset[random_idx]
    captions, attn_info = model.generate_caption(
        sample_image, idx2word, 
        return_attention=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3
    )
    save_attention_overlays(...)
    
    # 4. 체크포인트 저장
    if val_loss < best_val_loss:
        save_checkpoint(..., is_best=True)
```

**학습 전략**:
- **Encoder 활성화**: Encoder 파라미터도 함께 학습 (`requires_grad=True`)
- **Label Smoothing**: 과적합 방지 (smoothing=0.1)
- **Gradient Clipping**: 그래디언트 폭발 방지 (max_norm=5.0)
- **Teacher Forcing**: 학습 시 정답 캡션을 입력으로 사용

### Phase 5: 추론 과정 (generate_caption)

```
새로운 이미지 [1, 3, 224, 224]
    ↓
Encoder → (global_feat, spatial_feat, (H, W))
    ↓
초기 hidden state 계산 (global_feat 기반)
    ↓
for step in range(max_length):
    현재 단어 embedding
        ↓
    Attention (spatial_feat, hidden_state) → context
        ↓
    [word_embed; context] → LSTM → logits
        ↓
    반복 억제 적용 (repetition_penalty, no_repeat_ngram)
        ↓
    Greedy/Sampling으로 다음 단어 선택
        ↓
    <end> 토큰이면 종료
    ↓
토큰 인덱스 시퀀스
    ↓
Vocabulary.decode() → 단어 리스트
    ↓
생성된 캡션 문자열
```

**반복 억제 메커니즘**:
1. **N-gram 반복 방지**: 최근 N개 단어와 동일한 단어의 logit을 낮춤
2. **직전 토큰 억제**: 직전에 생성된 단어의 logit을 더 강하게 낮춤
3. **Repetition Penalty**: 반복된 토큰의 logit을 `repetition_penalty`로 나눔

### Phase 6: 평가 (evaluate_model)

```
테스트 데이터셋
    ↓
모델로 캡션 생성 (Greedy search)
    ↓
생성된 캡션 vs 참조 캡션
    ↓
modules/evaluation.py
    ↓
BLEU-1, BLEU-2, BLEU-3, BLEU-4
METEOR
ROUGE-L
CIDEr
```

## 3. 모듈 간 의존성 관계

```
main.py
    ├── models/image_caption_model.py
    │   ├── modules/resnet_18.py (ResNet)
    │   └── modules/decoder.py (CaptionDecoder)
    │       └── BahdanauAttention
    │
    ├── datasets/flickr8k.py
    │   └── Flickr8kImageCaptionDataset
    │
    ├── modules/evaluation.py
    │   └── calculate_bleu, calculate_meteor, ...
    │
    └── modules/attention_viz.py
        └── save_attention_overlays
```

## 4. 파일별 역할 요약

| 파일/모듈 | 역할 | 주요 클래스/함수 |
|----------|------|-----------------|
| `main.py` | 학습 스크립트 | `train_one_epoch()`, `validate()`, `evaluate_model()`, `LabelSmoothingCrossEntropy` |
| `models/image_caption_model.py` | 통합 모델 | `ImageCaptionModel` |
| `modules/resnet_18.py` | 이미지 인코더 | `ResNet`, `Stem`, `BasicBlock` |
| `modules/decoder.py` | 캡션 디코더 | `CaptionDecoder`, `BahdanauAttention`, `LSTMCell` |
| `modules/encoder.py` | 인코더 유틸리티 | `encode_images()` |
| `modules/evaluation.py` | 평가 지표 | `calculate_bleu()`, `calculate_meteor()`, `calculate_rouge()`, `calculate_cider()` |
| `modules/attention_viz.py` | Attention 시각화 | `save_attention_overlays()` |
| `modules/preprocess.py` | 전처리 유틸리티 | 전처리 함수들 |
| `datasets/flickr8k.py` | 데이터셋 로더 | `Flickr8kDataset`, `Flickr8kImageCaptionDataset` |

## 5. 실행 흐름 예시

### 학습 실행

```python
# 1. 데이터 전처리 (최초 1회만)
dataset = Flickr8kDataset()
dataset.load_captions_to_df()  # 단어장 구축

# 2. 데이터셋 생성
train_dataset = Flickr8kImageCaptionDataset(
    image_dir="datasets/data/Flickr8k_images",
    captions_file="datasets/data/captions_preprocessed/captions_padded.csv",
    transform=transform,
    split="train"
)

# 3. 모델 생성
encoder = ResNet(embed_size=256)
decoder = CaptionDecoder(
    embed_size=256,
    hidden_size=512,
    vocab_size=vocab_size,
    num_layers=2,
    dropout=0.1
)
model = ImageCaptionModel(encoder, decoder, vocab_size)

# 4. 학습
python main.py
```

### 추론 실행

```python
# 체크포인트 로드
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# 캡션 생성
image = Image.open("test_image.jpg")
transform = transforms.Compose([...])
image_tensor = transform(image).unsqueeze(0)

caption, attn_info = model.generate_caption(
    image_tensor,
    idx2word,
    max_length=20,
    return_attention=True,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3
)

print(caption[0])  # "a dog runs in the grass"
```

## 6. 핵심 설계 원칙

### 모듈화
- **Encoder/Decoder 분리**: 각각 독립적으로 수정 가능
- **Attention 메커니즘 분리**: `BahdanauAttention` 클래스로 독립 구현
- **평가 지표 분리**: `evaluation.py`에 모든 평가 함수 집중

### 재사용성
- **데이터셋 클래스**: 여러 데이터셋에 적용 가능한 구조
- **모델 아키텍처**: Encoder/Decoder를 교체하여 다른 모델 구성 가능

### 확장성
- **하이퍼파라미터**: `main.py`에서 중앙 관리
- **체크포인트 시스템**: 학습 중단 후 재개 가능
- **Attention 시각화**: 각 단어별 attention 분석 가능

### 추적성
- **로깅**: 학습/검증 손실 실시간 출력
- **체크포인트**: 최고 성능 모델 자동 저장
- **Attention heatmap**: 각 에폭마다 시각화 결과 저장

## 7. 주요 하이퍼파라미터

### 모델 하이퍼파라미터
- `embed_size`: 256 (인코더 출력 차원)
- `hidden_size`: 512 (LSTM hidden 차원)
- `num_layers`: 2 (LSTM 레이어 수)
- `dropout`: 0.1
- `vocab_size`: 단어장 크기 (동적)

### 학습 하이퍼파라미터
- `batch_size`: 128
- `num_epochs`: 25 (실제 학습) / 3 (빠른 테스트)
- `learning_rate`: 0.001
- `weight_decay`: 0.0001
- `gradient_clip`: 5.0
- `label_smoothing`: 0.1

### 추론 하이퍼파라미터
- `max_length`: 20 (최대 캡션 길이)
- `repetition_penalty`: 1.5 (반복 억제 강도)
- `no_repeat_ngram_size`: 3 (N-gram 반복 방지)
- `temperature`: 1.0 (샘플링 온도)
- `beam_size`: 1 (현재는 Greedy만 지원)

## 8. 데이터 구조

### 입력 데이터
- **이미지**: RGB 이미지, 224x224 크기로 리사이즈
- **캡션**: 최대 길이 20, 패딩 포함

### 출력 데이터
- **체크포인트**: 모델 가중치, 옵티마이저 상태, 에폭 번호, 손실
- **Attention heatmap**: 각 단어별 attention 가중치를 이미지에 오버레이
- **평가 결과**: BLEU, METEOR, ROUGE, CIDEr 점수

## 9. 주의사항 및 제한사항

### 현재 구현된 기능
- ✅ ResNet-18 기반 인코더
- ✅ Bahdanau Attention 메커니즘
- ✅ 다층 LSTM 디코더
- ✅ Label Smoothing
- ✅ 반복 억제 메커니즘
- ✅ Attention 시각화
- ✅ 다양한 평가 지표 (BLEU, METEOR, ROUGE, CIDEr)

### 미구현 기능
- ❌ Beam Search (현재는 Greedy만 지원)
- ❌ Transformer 기반 디코더
- ❌ 다른 인코더 (VGG, EfficientNet 등)
- ❌ MS COCO 데이터셋 지원

### 성능 최적화
- GPU 사용 시 `pin_memory=True`로 설정
- `num_workers=4`로 데이터 로딩 병렬화
- Gradient clipping으로 학습 안정화

## 10. 디버깅 및 모니터링

### 학습 모니터링
- **Top-k 예측 출력**: 각 배치마다 상위 k개 예측 단어 출력
- **단어 빈도 분석**: 데이터셋의 단어 빈도 통계
- **손실 추적**: 학습/검증 손실 실시간 출력

### Attention 분석
- **Heatmap 저장**: 각 에폭마다 샘플 이미지의 attention heatmap 저장
- **단어별 Attention**: 각 단어 생성 시 모델이 주목한 이미지 영역 시각화

### 문제 진단
- **붕괴 감지**: 같은 단어가 연속으로 3번 이상 반복되면 경고
- **Encoder 학습 확인**: Encoder 파라미터의 그래디언트 확인
- **파일 경로 검증**: 이미지 파일 존재 여부 확인
