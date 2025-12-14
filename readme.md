## 진행 순서

1. 🖼️ 데이터 수집 및 전처리
- 텍스트 전처리 (토큰화 진행 -> 벡터화 진행)
    - tokenizing
    - vocabulary (단어장 만들기)
- 이미지 전처리 
    - resizing
    - nomalization

2. 🤖 모델 설계 및 학습
Encoder & Decoder 2단계로 구분
- Encoder -> 이미지를 받아서 주요 특징을 추출 (벡터화 진행)
- Decoder -> 추출한 이미지의 특징을 바탕으로 문장 생성

3. 📝 평가 및 최적화
모델이 생성한 캡션에 대한 평가 진행 BLEU와 같은 평가 지표를 활용하여 score 계산

## 메인 로직 (main.py) 상세 설명

### 1. 초기 설정 및 하이퍼파라미터
```python
# 학습 모드 선택
FAST_TEST = False  # True: 빠른 테스트, False: 실제 학습

# 주요 하이퍼파라미터
- batch_size: 배치 크기 (128)
- num_epochs: 에폭 수 (10)
- hidden_size: LSTM hidden 크기 (512)
- embed_size: 임베딩 크기 (256)
- learning_rate: 학습률 (0.001)
- label_smoothing: Label Smoothing 비율 (0.1)
```

### 2. 데이터 로드
- **단어장 로드**: `word2idx.json`, `idx2word.json` 파일에서 단어-인덱스 매핑 로드
- **데이터셋 생성**: 
  - `Flickr8kImageCaptionDataset`으로 train/val 데이터셋 생성
  - 이미지: 224x224로 리사이즈 후 ImageNet 평균/표준편차로 정규화
  - 캡션: 패딩된 인덱스 시퀀스로 변환

### 3. 모델 구조
```
ImageCaptionModel
├── Encoder (ResNet)
│   └── 이미지 → 특징 벡터 (global + spatial features)
└── Decoder (CaptionDecoder)
    ├── LSTM (다층)
    ├── Bahdanau Attention (이미지 특징에 attention)
    └── Linear (단어 예측)
```

**특징:**
- Encoder는 **고정(freeze)**되어 학습되지 않음 (사전 학습된 ResNet 사용)
- Decoder만 학습됨
- Attention 메커니즘으로 이미지의 특정 영역에 집중

### 4. 학습 루프 (각 Epoch마다)

#### 4.1 학습 단계 (`train_one_epoch`)
```python
for batch in train_loader:
    1. Forward: model(images, captions) → logits
    2. Loss 계산: CrossEntropyLoss (Label Smoothing 적용)
    3. Backward: loss.backward()
    4. Gradient Clipping: max_norm=5.0
    5. Optimizer step: Adam optimizer
```

**특징:**
- Teacher Forcing 사용 (학습 시 정답 캡션을 입력으로 사용)
- Label Smoothing으로 과도한 확신 방지
- Gradient Clipping으로 그래디언트 폭발 방지

#### 4.2 검증 단계 (`validate`)
```python
for batch in val_loader:
    1. Forward: model(images, captions) → logits
    2. Loss 계산 (학습과 동일)
    3. 검증 손실 누적
```

#### 4.3 샘플 생성 및 시각화
각 epoch마다:
1. **캡션 생성**: 검증 데이터셋의 첫 번째 이미지로 캡션 생성
   - `generate_caption()` 메서드 사용
   - `return_attention=True`로 attention 정보도 함께 반환
   
2. **Attention Heatmap 저장**
   - 각 단어 생성 시 모델이 주목한 이미지 영역을 heatmap으로 시각화
   - 저장 위치: `checkpoints/attn_out/epoch_{번호}/`
   - 파일명: `{이미지명}_{단어번호}_{단어}.png`

3. **붕괴(Collapse) 검사**
   - 같은 단어가 연속 3번 이상 반복되는지 확인
   - 모델이 제대로 학습되고 있는지 모니터링

### 5. 체크포인트 저장
- **최고 성능 모델**: 검증 손실이 가장 낮을 때 `best_model.pth`로 저장
- **일반 체크포인트**: `checkpoint_epoch_{번호}.pth`로 저장
- 저장 내용: 모델 가중치, 옵티마이저 상태, 에폭, 손실

### 6. 주요 함수 설명

#### `train_one_epoch()`
- 한 epoch 동안 학습 수행
- 진행 상황을 tqdm progress bar로 표시
- 옵션으로 top-k 예측 출력 가능

#### `validate()`
- 검증 데이터셋으로 모델 성능 평가
- 학습 없이 forward pass만 수행

#### `generate_caption()`
- 추론 시 사용: 이미지로부터 캡션 생성
- Greedy search 또는 Sampling 방식 지원
- `return_attention=True` 시 attention 가중치도 반환

#### `save_attention_overlays()`
- Attention heatmap을 원본 이미지 위에 overlay하여 저장
- 각 단어별로 어떤 이미지 영역에 주목했는지 시각화

### 7. 학습 전략
- **Encoder Freeze**: 사전 학습된 ResNet은 고정, Decoder만 학습
- **Label Smoothing**: 과적합 방지 및 일반화 성능 향상
- **Gradient Clipping**: 그래디언트 폭발 방지
- **반복 억제**: 같은 단어가 연속으로 반복되는 것을 방지

## 테스트 데이터 확보
- Flickr8k 추가 사용(용량이 적어서 테스트로 돌리기 더 좋아보임)
    - https://www.kaggle.com/datasets/adityajn105/flickr8k
- 데이터가 부족한 경우 kaggle에서 제공하는 MS COCO dataset 추가 활용
    - (https://www.kaggle.com/datasets?search=MS+COCO)

