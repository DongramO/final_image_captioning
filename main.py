"""
이미지 캡셔닝 모델 학습 스크립트
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import time
from tqdm import tqdm

import modules
from modules.decoder import CaptionDecoder
from models.image_caption_model import ImageCaptionModel
from datasets.flickr8k import Flickr8kImageCaptionDataset
from visual import visualize_predictions

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, print_every=100):
    """
    한 에폭 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        epoch: 현재 에폭 번호
        print_every: 몇 iteration마다 로그 출력
        
    Returns:
        avg_loss: 평균 손실
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, captions) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        
        # Forward pass
        outputs = model(images, captions)  # [batch_size, seq_length-1, vocab_size]
        
        # 출력과 타겟 정렬
        # outputs: [batch_size, seq_length-1, vocab_size]
        # targets: [batch_size, seq_length-1] (captions의 <start> 다음부터)
        targets = captions[:, 1:]  # <start> 토큰 제외
        
        # Reshape for loss calculation
        batch_size, seq_length, vocab_size = outputs.shape
        outputs = outputs.reshape(-1, vocab_size)  # [batch_size * seq_length, vocab_size]
        targets = targets.reshape(-1)  # [batch_size * seq_length]
        
        # Loss 계산
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
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


def validate(model, dataloader, criterion, device):
    """
    검증
    
    Args:
        model: 검증할 모델
        dataloader: 검증 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        
    Returns:
        avg_loss: 평균 손실
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(images, captions)
            
            # 출력과 타겟 정렬
            targets = captions[:, 1:]
            
            # Reshape for loss calculation
            batch_size, seq_length, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            
            # Loss 계산
            loss = criterion(outputs, targets)
            
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


def main():
    # 설정
    _ROOT = os.path.dirname(os.path.dirname(__file__))
    
    # 하이퍼파라미터
    FAST_TEST = True  # True로 설정하면 빠른 테스트 모드
    
    if FAST_TEST:
        # 빠른 테스트 설정
        batch_size = 64
        num_epochs = 3
        max_train_samples = 1000
        max_val_samples = 200
        validate_every = 1
        hidden_size = 256
        embed_size = 128
        num_layers = 1
    else:
        # 실제 학습 설정
        batch_size = 128
        num_epochs = 20
        max_train_samples = None
        max_val_samples = None
        validate_every = 1
        hidden_size = 512
        embed_size = 256
        num_layers = 2
    
    dropout = 0.5
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

    # 경로 설정
    image_dir = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "Flickr8k_images")
    captions_file = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "captions_preprocessed", "captions_padded.csv")
    word2idx_path = os.path.join(_ROOT, "final_image_captioning", "datasets", "data","captions_preprocessed", "word2idx.json")
    checkpoint_dir = os.path.join(_ROOT, "final_image_captioning", "checkpoints")
    
    # 단어장 로드
    print("\n단어장 로드 중...")
    with open(word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)
    print(f"단어장 크기: {vocab_size}")
    
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
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if not FAST_TEST else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if not FAST_TEST else 0,
        pin_memory=True if torch.cuda.is_available() else False
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
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx.get('<pad>', 0))
    optimizer = optim.Adam(
        model.parameters(),
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
            model, train_loader, criterion, optimizer, device, epoch, print_every
        )
        
        # 검증
        val_loss = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # 결과 출력
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  학습 손실: {train_loss:.4f}")
        print(f"  검증 손실: {val_loss:.4f}")
        print(f"  소요 시간: {epoch_time:.2f}초")
        
        # 최고 성능 모델 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  ✓ 새로운 최고 성능! (검증 손실: {val_loss:.4f})")
        
        # 체크포인트 저장 조건 개선
        should_save = False
        if (epoch + 1) % save_every == 0:
            should_save = True
            print(f"  ✓ 주기적 체크포인트 저장 (에폭 {epoch+1})")
        elif is_best:
            should_save = True
            print(f"  ✓ 최고 성능 모델 저장")
        elif epoch + 1 == num_epochs:  # 마지막 에폭에서도 저장
            should_save = True
            print(f"  ✓ 마지막 에폭 체크포인트 저장")
        
        if should_save:
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_dir, is_best)
            print(f"  ✓ 체크포인트 저장 완료: {checkpoint_dir}")
        
        print("-" * 50)


if __name__ == '__main__':
    main()