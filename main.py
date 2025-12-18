"""
이미지 캡셔닝 모델 학습 스크립트
"""
from PIL import Image
from modules.attention_viz import save_attention_overlays
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import time
import random
from tqdm import tqdm

import modules
from modules.decoder import CaptionDecoder
from models.image_caption_model import ImageCaptionModel
from datasets.flickr8k import Flickr8kImageCaptionDataset
import torch.nn.functional as F

def get_topk_predictions(logits, idx2word, k=5):
    """
    로짓에서 top-k 예측 반환
    
    Args:
        logits: [vocab_size] 또는 [1, vocab_size] 텐서
        idx2word: 인덱스-단어 매핑 딕셔너리
        k: 상위 k개
        
    Returns:
        List of (word, index, probability) tuples
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # 확률로 변환
    probs = torch.softmax(logits, dim=-1)  # [1, vocab_size]
    
    # Top-k
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    
    results = []
    # idx2word 키 타입 확인 (문자열 또는 정수)
    if idx2word and len(idx2word) > 0:
        first_key = list(idx2word.keys())[0]
        key_is_str = isinstance(first_key, str)
    else:
        key_is_str = False
    
    for i in range(k):
        idx = int(topk_indices[0, i].item())
        prob = float(topk_probs[0, i].item())
        
        # 키 타입에 따라 접근
        if key_is_str:
            word = idx2word.get(str(idx), '<unk>')
        else:
            word = idx2word.get(idx, '<unk>')
        
        results.append((word, idx, prob))
    
    return results

def print_topk_predictions(logits, idx2word, k=5, prefix=""):
    """
    로짓의 top-k 예측을 출력
    
    Args:
        logits: [vocab_size] 또는 [1, vocab_size] 텐서
        idx2word: 인덱스-단어 매핑 딕셔너리
        k: 상위 k개
        prefix: 출력 앞에 붙일 문자열
    """
    topk_preds = get_topk_predictions(logits, idx2word, k)
    print(f"{prefix}Top-{k} 예측:")
    for i, (word, idx, prob) in enumerate(topk_preds, 1):
        print(f"  {i}. {word:15s} (idx: {idx:5d}, prob: {prob:.4f})")

def analyze_word_frequency(dataset, idx2word, top_n=20):
    """
    데이터셋에서 단어 빈도 분석
    
    Args:
        dataset: 데이터셋
        idx2word: 인덱스-단어 매핑 딕셔너리
        top_n: 상위 n개 출력
    """
    from collections import Counter
    
    word_counts = Counter()
    total_words = 0
    
    for data_pair in dataset.data_pairs:
        caption = data_pair['caption']
        for idx in caption:
            # 특수 토큰 제외
            if isinstance(idx, (int, str)):
                idx_int = int(idx) if isinstance(idx, str) else idx
                
                # idx2word에서 단어 찾기
                word = idx2word.get(idx_int, None)
                if word and word not in ['<pad>', '<start>', '<end>', '<unk>']:
                    word_counts[word] += 1
                    total_words += 1
    
    print(f"\n=== 단어 빈도 분석 (총 {total_words}개 단어) ===")
    print(f"상위 {top_n}개 단어:")
    for i, (word, count) in enumerate(word_counts.most_common(top_n), 1):
        freq = count / total_words * 100
        print(f"  {i:2d}. {word:15s}: {count:6d}회 ({freq:5.2f}%)")
    
    return word_counts

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: [N, V], target: [N]
        V = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # NLL loss (정답 토큰)
        nll = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.ignore_index)

        # smooth loss (전체 토큰 평균)
        smooth = -log_probs.mean(dim=-1)  # [N]

        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            nll = nll[mask]
            smooth = smooth[mask]

        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, print_every=100, idx2word=None, show_topk=False, topk=10):
    """
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        epoch: 현재 에폭 번호
        print_every: 몇 iteration마다 로그 출력
        idx2word: 인덱스-단어 매핑 (top-k 출력용)
        show_topk: top-k 출력 여부
        topk: 출력할 top-k 개수
        
    Returns:
        avg_loss: 평균 손실
    """
    model.train()  # 전체 모델을 학습 모드로 (encoder도 포함)
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, captions) in enumerate(progress_bar):
        # non_blocking=True: CPU-GPU 전송을 비동기로 처리 (데이터 로딩과 병렬)
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images, captions)
        targets = captions[:, 1:]


        # Top-k 예측 출력 (옵션, reshape 전에 수행)
        if show_topk and idx2word is not None and (batch_idx + 1) % print_every == 0:
            if outputs.shape[1] > 0:  # 시퀀스가 있으면
                first_step_logits = outputs[0, 0, :].detach()  # 첫 번째 배치, 첫 번째 시퀀스 위치
                print(f"\n[Epoch {epoch}, Batch {batch_idx+1}]")
                print_topk_predictions(first_step_logits, idx2word, k=topk, prefix="")
        
        # Reshape for loss calculation
        batch_size, seq_length, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)  # [batch_size * seq_length, vocab_size]
        targets_flat = targets.reshape(-1)  # [batch_size * seq_length]
        
        # Loss 계산
        loss = criterion(outputs_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑 (전체 모델 파라미터에 대해)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # 가중치 업데이트
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item()
        num_batches += 1
        
        # 진행 상황 업데이트
        if (batch_idx + 1) % print_every == 0:
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, idx2word=None, show_topk=False, topk=10):
    """
    검증
    
    Args:
        model: 검증할 모델
        dataloader: 검증 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        idx2word: 인덱스-단어 매핑 (top-k 출력용)
        show_topk: top-k 출력 여부
        topk: 출력할 top-k 개수
        
    Returns:
        avg_loss: 평균 손실
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images, captions)
            
            # 출력과 타겟 정렬
            targets = captions[:, 1:]
            
            # Top-k 예측 출력 (첫 번째 배치에서만)
            if show_topk and idx2word is not None and batch_idx == 0:
                if outputs.shape[1] > 0:  # 시퀀스가 있으면
                    first_step_logits = outputs[0, 0, :]  # 첫 번째 배치, 첫 번째 시퀀스 위치
                    print("\n[Validation]")
                    print_topk_predictions(first_step_logits, idx2word, k=topk, prefix="")
            
            # Reshape for loss calculation
            batch_size, seq_length, vocab_size = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Loss 계산
            loss = criterion(outputs_flat, targets_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """
    체크포인트 저장
    
    Args:
        model: 저장할 모델
        optimizer: 옵티마이저
        epoch: 에폭 번호
        loss: 손실 값
        checkpoint_dir: 체크포인트 저장 디렉토리
        is_best: 최고 성능 모델인지 여부
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # 일반 체크포인트
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 최고 성능 모델
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 최고 성능 모델 저장: {best_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    체크포인트 로드
    
    Args:
        model: 모델
        optimizer: 옵티마이저
        checkpoint_path: 체크포인트 경로
        device: 디바이스
        
    Returns:
        epoch: 시작 에폭 번호
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"체크포인트 로드: {checkpoint_path}")
    print(f"  에폭: {epoch}, 손실: {loss:.4f}")
    
    return epoch + 1

def evaluate_model(model, dataloader, device, idx2word, word2idx, max_length=20, beam_size=1):
    """
    모델 평가 (BLEU, METEOR, ROUGE, CIDEr 점수 계산)
    
    Args:
        model: 평가할 모델
        dataloader: 평가 데이터 로더
        device: 디바이스
        idx2word: 인덱스-단어 매핑 딕셔너리
        word2idx: 단어-인덱스 매핑 딕셔너리
        max_length: 최대 생성 길이
        beam_size: Beam search 크기
        
    Returns:
        metrics: 평가 지표 딕셔너리
    """
    from modules.evaluation import calculate_bleu, calculate_meteor, calculate_rouge, calculate_cider
    
    model.eval()
    
    # 토큰 인덱스
    start_token = word2idx.get('<start>', 1)
    end_token = word2idx.get('<end>', 2)
    
    all_references = []  # 참조 캡션 (각 이미지당 여러 개)
    all_candidates = []  # 생성된 캡션
    image_names = []  # 이미지 이름 (디버깅용)
    
    print("\n캡션 생성 중...")
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device, non_blocking=True)
            
            # 캡션 생성
            result = model.generate_caption(
                images,
                idx2word,
                max_length=20,
                start_token=start_token,
                end_token=end_token,
                beam_size=1,
                show_topk=True,
                temperature=1.0,
                sample=False,      # Greedy search 사용
                topk=10,
                return_attention=True,
                repetition_penalty=1.5,  # 반복 억제 강도
                no_repeat_ngram_size=3,  # 최근 3개 단어 고려
                use_topk_sampling=False,  # 순수 greedy search (argmax)
            )
            
            # return_attention=True일 때 튜플 반환 처리
            if isinstance(result, tuple):
                generated_captions, _ = result  # attn_info는 사용하지 않음
            else:
                generated_captions = result
            
            # 배치 내 각 샘플 처리
            for i in range(len(generated_captions)):
                # 생성된 캡션을 단어 리스트로 변환 (이미 문자열이므로 split)
                gen_caption = generated_captions[i].split()
                all_candidates.append(gen_caption)
                
                # 참조 캡션 수집 (같은 이미지의 다른 캡션들)
                # 데이터셋에서 이미지 이름으로 참조 캡션 찾기
                dataset = dataloader.dataset
                current_idx = batch_idx * dataloader.batch_size + i
                
                if current_idx < len(dataset):
                    image_name = dataset.get_image_name(current_idx)
                    image_names.append(image_name)
                    
                    # 같은 이미지의 모든 캡션 찾기
                    ref_captions = []
                    for data_pair in dataset.data_pairs:
                        if data_pair['image_name'] == image_name:
                            # 인덱스를 단어로 변환
                            caption_indices = data_pair['caption']
                            words = []
                            for idx in caption_indices:
                                idx = int(idx)
                                if idx == end_token:
                                    break
                                if idx == start_token:
                                    continue
                                if idx in idx2word:
                                    word = idx2word[idx]
                                    if word not in ['<pad>', '<start>', '<end>', '<unk>']:
                                        words.append(word)
                            if words:  # 빈 캡션 제외
                                ref_captions.append(words)
                    
                    # 참조 캡션이 없으면 현재 캡션만 사용
                    if not ref_captions:
                        # 현재 배치의 캡션을 참조로 사용
                        caption_indices = captions[i].cpu().tolist()
                        words = []
                        for idx in caption_indices:
                            idx = int(idx)
                            if idx == end_token:
                                break
                            if idx == start_token:
                                continue
                            if idx in idx2word:
                                word = idx2word[idx]
                                if word not in ['<pad>', '<start>', '<end>', '<unk>']:
                                    words.append(word)
                        if words:
                            ref_captions.append(words)
                    
                    all_references.append(ref_captions if ref_captions else [[]])
                else:
                    all_references.append([[]])
    
    print(f"\n총 {len(all_candidates)}개 캡션 평가 중...")
    
    # 평가 지표 계산
    metrics = {}
    
    # BLEU 점수
    print("BLEU 점수 계산 중...")
    bleu_scores = calculate_bleu(all_references, all_candidates, n=4)
    metrics.update(bleu_scores)
    
    # METEOR 점수
    print("METEOR 점수 계산 중...")
    meteor_score = calculate_meteor(all_references, all_candidates)
    metrics['METEOR'] = meteor_score
    
    # ROUGE 점수
    print("ROUGE 점수 계산 중...")
    rouge_scores = calculate_rouge(all_references, all_candidates)
    metrics.update(rouge_scores)
    
    # CIDEr 점수
    print("CIDEr 점수 계산 중...")
    cider_score = calculate_cider(all_references, all_candidates)
    metrics['CIDEr'] = cider_score
    
    return metrics, all_references, all_candidates, image_names

def main():
    # 설정
    # 현재 파일의 디렉토리를 기준으로 프로젝트 루트 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트는 main.py가 있는 디렉토리
    _ROOT = current_dir
    
    # 하이퍼파라미터
    FAST_TEST = False  # True로 설정하면 빠른 테스트 모드
    
    if FAST_TEST:
        # 빠른 테스트 설정
        batch_size = 256
        num_epochs = 3
        max_train_samples = 1000
        max_val_samples = 200
        validate_every = 1
        hidden_size = 512
        embed_size = 128
        num_layers = 1
    else:
        # 실제 학습 설정
        batch_size = 256
        num_epochs = 25
        max_train_samples = None
        max_val_samples = None
        validate_every = 1
        hidden_size = 512
        embed_size = 256
        num_layers = 2
    
    dropout = 0.1
    learning_rate = 0.001
    weight_decay = 0.0001
    gradient_clip = 5.0
    save_every = 5
    print_every = 100
    
    # save_every를 num_epochs에 맞게 조정
    if FAST_TEST:
        save_every = 1  # 빠른 테스트 시 매 에폭마다 저장
    else:
        save_every = 5  # 실제 학습 시 5 에폭마다 저장

    # 경로 설정 (프로젝트 루트 기준)
    image_dir = os.path.join(_ROOT, "datasets", "data", "Flickr8k_images")
    captions_file = os.path.join(_ROOT, "datasets", "data", "captions_preprocessed", "captions_padded.csv")
    word2idx_path = os.path.join(_ROOT, "datasets", "data", "captions_preprocessed", "word2idx.json")
    checkpoint_dir = os.path.join(_ROOT, "checkpoints")
    
    # 단어장 로드
    with open(word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)
    print(f"단어장 크기: {vocab_size}")
    
    # idx2word 딕셔너리 생성 (top-k 출력용)
    idx2word_path = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "captions_preprocessed", "idx2word.json")
    if os.path.exists(idx2word_path):
        with open(idx2word_path, 'r', encoding='utf-8') as f:
            idx2word = json.load(f)
            idx2word = {int(k): v for k, v in idx2word.items()}
    else:
        idx2word = {v: k for k, v in word2idx.items()}
    
    # top-k 출력 설정
    show_topk = True  # top-k 출력 여부
    topk = 5  # 출력할 top-k 개수
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Transform 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    train_dataset = Flickr8kImageCaptionDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        transform=transform,
        split="train"
    )
    
    val_dataset = Flickr8kImageCaptionDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        transform=transform,
        split="val"
    )
    
    if max_train_samples:
        train_dataset.data_pairs = train_dataset.data_pairs[:max_train_samples]
        print(f"⚠ 빠른 테스트: 학습 데이터를 {max_train_samples}개로 제한")
    
    if max_val_samples:
        val_dataset.data_pairs = val_dataset.data_pairs[:max_val_samples]
        print(f"⚠ 빠른 테스트: 검증 데이터를 {max_val_samples}개로 제한")
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 단어 빈도 분석 (옵션)
    analyze_frequency = True
    if analyze_frequency:
        analyze_word_frequency(train_dataset, idx2word, top_n=20)
    
    # GPU 설정 확인 및 최적화
    if torch.cuda.is_available():
        print(f"\n✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # cuDNN 벤치마크 활성화 (입력 크기가 고정된 경우 성능 향상)
        torch.backends.cudnn.benchmark = True
    else:
        print("\n⚠ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        print("  GPU를 사용하려면 CUDA가 설치된 PyTorch를 설치하세요:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    
    # DataLoader 생성 (GPU 최적화)
    num_workers = 8 if torch.cuda.is_available() and not FAST_TEST else (4 if not FAST_TEST else 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 and not FAST_TEST else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 and not FAST_TEST else False
    )
    
    # 모델 생성
    print("\n모델 생성 중...")
    encoder = modules.ResNet(embed_size=embed_size)
    decoder = CaptionDecoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model = ImageCaptionModel(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size
    )
    model = model.to(device)
    
    # 파라미터 개수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    
    # 손실 함수 및 옵티마이저
    # Label Smoothing 사용 여부 (과도한 확신 방지)
    use_label_smoothing = True
    label_smoothing = 0.1  # 0.1 = 10% smoothing (일반적으로 0.1~0.2 사용)
    
    pad_idx = word2idx.get('<pad>', 0)
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            ignore_index=pad_idx
        )
        print(f"\n✓ Label Smoothing 적용 (smoothing={label_smoothing})")
        print("  → 모델이 특정 단어에 과도하게 확신하는 것을 방지합니다")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        print(f"\n✓ 일반 CrossEntropyLoss 사용")
    
    # Encoder 활성화: 이미지 특징 추출도 학습 가능하도록 설정
    # encoder freeze 제거 - encoder도 함께 학습
    for p in model.encoder.parameters():
        p.requires_grad = True  # Encoder 파라미터도 학습 가능
    
    print("\n✓ Encoder 활성화: 이미지 특징 추출도 학습됩니다")

    # 전체 모델 파라미터를 optimizer에 포함
    optimizer = optim.Adam(
        model.parameters(),  # encoder + decoder 모두 학습
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 학습 시작
    print("\n" + "="*50)
    print("학습 시작")
    print("="*50)
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # 학습
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, print_every,
            idx2word=idx2word if show_topk else None,
            show_topk=show_topk,
            topk=topk
        )
        
        # 검증
        val_loss = validate(
            model, val_loader, criterion, device,
            idx2word=idx2word if show_topk else None,
            show_topk=show_topk,
            topk=topk
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # 결과 출력
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  학습 손실: {train_loss:.4f}")
        print(f"  검증 손실: {val_loss:.4f}")
        print(f"  소요 시간: {epoch_time:.2f}초")
        
        # 매 epoch마다 greedy 생성 결과 출력 (하나의 이미지만 처리)
        print("\n" + "-"*50)
        print(f"[Epoch {epoch+1}] Greedy 생성 결과 (attention heatmap):")
        print("-"*50)
        
        #검증 진행을 위해 model eval하고 with torch.no_grad() 사용
        model.eval()
        with torch.no_grad():
            # 검증 데이터셋에서 랜덤하게 하나의 이미지 선택
            val_dataset_size = len(val_dataset)
            sample_idx = random.randint(0, val_dataset_size - 1)
            
            # Greedy 생성 (top-k 출력 포함)
            start_token = word2idx.get('<start>', 1)
            end_token = word2idx.get('<end>', 2)
            
            out_dir = os.path.join(checkpoint_dir, "attn_out", f"epoch_{epoch+1}")
            
            # 검증 데이터셋에서 이미지 가져오기
            sample_image, _ = val_dataset[sample_idx]
            sample_image = sample_image.unsqueeze(0).to(device, non_blocking=True)  # [1, C, H, W]
            
            generated_captions, attn_info = model.generate_caption(
                sample_image,
                idx2word,
                max_length=20,
                start_token=start_token,
                end_token=end_token,
                beam_size=1,
                show_topk=True,  # top-k 출력
                temperature=1.0,
                sample=False,  # Greedy search 사용
                topk=10,
                return_attention=True,
                repetition_penalty=1.5,  # 반복 억제 강도
                no_repeat_ngram_size=3,  # 최근 3개 단어 고려
                use_topk_sampling=False,  # 순수 greedy search (argmax)
            )

            generated_caption = generated_captions[0]
            sample_name = val_dataset.get_image_name(sample_idx)
            
            print(f"\n이미지: {sample_name}")
            print(f"  생성된 캡션: {generated_caption}")

            raw_path = os.path.join(image_dir, sample_name)
            img_pil = Image.open(raw_path).convert("RGB").resize((224, 224))

            prefix = os.path.splitext(sample_name)[0]

            save_attention_overlays(
                image=img_pil,
                words=attn_info[0]["words"],
                alphas=attn_info[0]["alphas"],
                spatial_hw=attn_info[0]["spatial_hw"],   # 보통 (7,7)
                out_dir=out_dir,
                prefix=prefix,
            )
            
            # 붕괴 여부 확인
            words = generated_caption.split()
            if len(words) >= 2:
                # 같은 단어가 연속으로 3번 이상 나오면 붕괴
                consecutive_same = False
                for i in range(len(words) - 2):
                    if words[i] == words[i+1] == words[i+2]:
                        consecutive_same = True
                        break
                
                if consecutive_same:
                    print("  ⚠️  경고: 같은 단어가 연속으로 반복됩니다 (붕괴 가능성)")
                else:
                    print("  ✓ 정상: 다양한 단어가 생성되었습니다")
            else:
                print("  ⚠️  경고: 생성된 단어가 너무 적습니다")
            
            print(f"\n[Attention] heatmap 저장 완료: {out_dir}")
        
        should_save = True

        # 최고 성능 모델 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  ✓ 새로운 최고 성능! (검증 손실: {val_loss:.4f})")
        elif epoch + 1 == num_epochs:  # 마지막 에폭에서도 저장
            should_save = True
            print(f"  ✓ 마지막 에폭 체크포인트 저장")
        
        if should_save:
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_dir, is_best)
            print(f"  ✓ 체크포인트 저장 완료: {checkpoint_dir}")
        
        print("-" * 50)
        

    print("\n" + "="*50)
    print("테스트 데이터셋 평가")
    print("="*50)
    
    enc_first = next(model.encoder.parameters())
    dec_first = next(model.decoder.parameters())
    print(enc_first.grad)
    print(dec_first.grad is None)

    w0 = model.encoder.stem.conv1.weight.detach().clone()
    w1 = model.encoder.stem.conv1.weight.detach().clone()
    # print(torch.allclose(w0, w1))  # True면 encoder가 안 바뀜


    test_dataset = Flickr8kImageCaptionDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        transform=transform,
        split="test"
    )
    
    num_workers = 8 if torch.cuda.is_available() and not FAST_TEST else (4 if not FAST_TEST else 0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 and not FAST_TEST else False
    )
    
    # 평가 실행
    metrics, references, candidates, image_names = evaluate_model(
        model, test_loader, device, idx2word, word2idx, 
        max_length=50, beam_size=1
    )
    
    # 결과 출력
    print("\n" + "="*50)
    print("평가 결과")
    print("="*50)
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()