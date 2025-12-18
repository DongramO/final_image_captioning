import os
import torch
import torch.nn as nn
from torchvision import transforms
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import glob
import textwrap

import modules
from modules.decoder import CaptionDecoder
from models.image_caption_model import ImageCaptionModel
from datasets.flickr8k import Flickr8kImageCaptionDataset
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    정규화된 이미지 텐서를 원래 형태로 복원
    
    Args:
        tensor: 정규화된 이미지 텐서 [C, H, W]
        mean: 정규화에 사용된 평균
        std: 정규화에 사용된 표준편차
        
    Returns:
        numpy array: [H, W, C] 형태의 이미지 (0-255 범위)
    """
    img = tensor.cpu().clone()
    
    # Denormalize
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    # 0-1 범위로 클리핑
    img = torch.clamp(img, 0, 1)
    
    # [C, H, W] -> [H, W, C]
    img = img.permute(1, 2, 0).numpy()
    
    # 0-255 범위로 변환
    img = (img * 255).astype(np.uint8)
    
    return img


def visualize_predictions(
    model, 
    test_dataset, 
    idx2word, 
    device, 
    num_samples=5,
    start_token=1,
    end_token=2
):
    """
    테스트 셋의 이미지와 예측 캡션을 시각화 (개선된 레이아웃)
    
    Args:
        model: 학습된 모델
        test_dataset: 테스트 데이터셋
        idx2word: 인덱스를 단어로 변환하는 딕셔너리
        device: 디바이스
        num_samples: 시각화할 샘플 개수
        start_token: 시작 토큰 인덱스
        end_token: 종료 토큰 인덱스
    """
    model.eval()
    
    # 랜덤 샘플 선택
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    # 개선된 레이아웃: 각 샘플당 이미지와 텍스트 영역을 분리
    fig = plt.figure(figsize=(14, 5 * num_samples))
    gs = fig.add_gridspec(num_samples, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.2)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 이미지와 실제 캡션 로드
            image, true_caption = test_dataset[idx]
            image_name = test_dataset.get_image_name(idx)
            
            # 이미지를 배치로 변환
            image_batch = image.unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 캡션 생성
            result = model.generate_caption(
                image_batch,
                idx2word,
                max_length=20,
                start_token=start_token,
                end_token=end_token,
                beam_size=1,
                temperature=1.0,
                sample=False,  # Greedy search 사용
                return_attention=True,
                repetition_penalty=1.5,  # 반복 억제 강도
                no_repeat_ngram_size=3,  # 최근 3개 단어 고려
                use_topk_sampling=False,  # 순수 greedy search (argmax)
            )

            # 반환값 처리
            if isinstance(result, tuple):
                predicted_captions, attn_info = result
                predicted_caption = predicted_captions[0]
            else:
                predicted_captions = result
                predicted_caption = predicted_captions[0]
                attn_info = None

            # 실제 캡션 변환 (인덱스 -> 단어) - 먼저 처리
            true_caption_words = []
            for token_idx in true_caption.cpu().tolist():
                if token_idx == end_token:
                    break
                if token_idx == start_token:
                    continue
                if token_idx in idx2word:
                    word = idx2word[token_idx]
                    if word not in ['<pad>', '<start>', '<end>', '<unk>']:
                        true_caption_words.append(word)
            true_caption_str = ' '.join(true_caption_words)

            # 디버깅: 각 이미지마다 다른 캡션이 생성되는지 확인
            print(f"\n[샘플 {i+1}] 이미지: {image_name}")
            print(f"  예측 캡션: {predicted_caption}")
            print(f"  실제 캡션: {true_caption_str}")
            
            # 이미지 denormalize
            img_display = denormalize_image(image)
            
            # 이미지 영역
            ax_img = fig.add_subplot(gs[i, 0])
            ax_img.imshow(img_display)
            ax_img.axis('off')
            ax_img.set_title(f"샘플 {i+1}: {image_name}", fontsize=12, fontweight='bold', pad=10)
            
            # 텍스트 영역
            ax_text = fig.add_subplot(gs[i, 1])
            ax_text.axis('off')
            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(0, 1)
            
            # 텍스트 줄바꿈 (최대 40자로 줄여서 공간 확보)
            wrapped_true = textwrap.fill(true_caption_str, width=40)
            wrapped_pred = textwrap.fill(predicted_caption, width=40)
            
            # 실제 캡션 (녹색) - 상단
            y_label_top = 0.95
            y_text_top = 0.80
            
            # 라벨 텍스트 (x 좌표를 0.01로 조정하여 여백 확보)
            ax_text.text(0.01, y_label_top, "실제 캡션 (Ground Truth):", 
                        fontsize=11, fontweight='bold', color='#2e7d32',
                        transform=ax_text.transAxes, verticalalignment='top',
                        horizontalalignment='left', family='Malgun Gothic',
                        clip_on=False)
            
            # 실제 캡션 텍스트 박스
            true_lines = wrapped_true.split('\n')
            true_height = max(0.12, len(true_lines) * 0.04)
            ax_text.text(0.01, y_text_top, wrapped_true,
                        fontsize=10, color='#1b5e20',
                        transform=ax_text.transAxes, verticalalignment='top',
                        horizontalalignment='left', family='Malgun Gothic',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f5e9', 
                                edgecolor='#2e7d32', linewidth=1.5, alpha=0.8),
                        clip_on=False)
            
            # 구분선 (axes 좌표계 사용)
            ax_text.plot([0.01, 0.99], [0.5, 0.5], color='gray', linestyle='--', 
                        linewidth=1, alpha=0.6, transform=ax_text.transAxes,
                        clip_on=False)
            
            # 예측 캡션 (파란색) - 하단
            y_label_bottom = 0.45
            y_text_bottom = 0.30
            
            # 라벨 텍스트
            ax_text.text(0.01, y_label_bottom, "예측 캡션 (Prediction):", 
                        fontsize=11, fontweight='bold', color='#1565c0',
                        transform=ax_text.transAxes, verticalalignment='top',
                        horizontalalignment='left', family='Malgun Gothic',
                        clip_on=False)
            
            # 예측 캡션 텍스트 박스
            pred_lines = wrapped_pred.split('\n')
            pred_height = max(0.12, len(pred_lines) * 0.04)
            ax_text.text(0.01, y_text_bottom, wrapped_pred,
                        fontsize=10, color='#0d47a1',
                        transform=ax_text.transAxes, verticalalignment='top',
                        horizontalalignment='left', family='Malgun Gothic',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#e3f2fd', 
                                edgecolor='#1565c0', linewidth=1.5, alpha=0.8),
                        clip_on=False)
    
    plt.suptitle('이미지 캡셔닝 예측 결과', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('test_predictions.png', dpi=200, bbox_inches='tight', facecolor='white')
    print("시각화 저장 완료: test_predictions.png")
    plt.show()


def load_model(checkpoint_path, vocab_size, device):
    """
    체크포인트에서 모델 로드 (설정 자동 추론)
    """
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 체크포인트에서 실제 설정 추론
    if 'decoder.init_h.weight' in state_dict:
        embed_size = state_dict['decoder.init_h.weight'].shape[1]
        hidden_size = state_dict['decoder.init_h.weight'].shape[0]
    else:
        embed_size = checkpoint.get('embed_size', 128)
        hidden_size = checkpoint.get('hidden_size', 256)
    
    # vocab_size는 체크포인트의 것을 사용
    if 'decoder.embedding.weight' in state_dict:
        checkpoint_vocab_size = state_dict['decoder.embedding.weight'].shape[0]
        if checkpoint_vocab_size != vocab_size:
            print(f"⚠️ vocab_size 불일치: 체크포인트({checkpoint_vocab_size}) vs 현재({vocab_size})")
            print(f"   체크포인트의 vocab_size({checkpoint_vocab_size})를 사용합니다.")
            vocab_size = checkpoint_vocab_size
    
    # num_layers 추론
    num_layers = 1
    for i in range(10):
        if f'decoder.lstm_cells.{i}.W_f.weight' in state_dict:
            num_layers = i + 1
        else:
            break
    
    print(f"체크포인트에서 추론한 설정:")
    print(f"  embed_size: {embed_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  num_layers: {num_layers}")
    
    # 모델 생성
    encoder = modules.ResNet(embed_size=embed_size)
    decoder = CaptionDecoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=0.1
    )
    
    model = ImageCaptionModel(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size
    )
    
    # 사이즈가 맞는 레이어만 로드
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"⚠️ 사이즈 불일치로 제외: {key}")
        else:
            print(f"⚠️ 키 없음으로 제외: {key}")
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"모델 로드 완료: {checkpoint_path}")
    print(f"  에폭: {checkpoint['epoch']}, 손실: {checkpoint['loss']:.4f}")
    print(f"  로드된 레이어: {len(filtered_state_dict)}/{len(state_dict)}")
    
    return model


def main():
    # 설정
    # 현재 파일의 디렉토리를 기준으로 프로젝트 루트 찾기 (main.py와 동일)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트는 visual.py가 있는 디렉토리 (final_image_captioning)
    _ROOT = current_dir
    
    # 경로 설정
    image_dir = os.path.join(_ROOT, "datasets", "data", "Flickr8k_images")
    captions_file = os.path.join(_ROOT, "datasets", "data", "captions_preprocessed", "captions_padded.csv")
    word2idx_path = os.path.join(_ROOT, "datasets", "data", "captions_preprocessed", "word2idx.json")
    idx2word_path = os.path.join(_ROOT, "datasets", "data", "captions_preprocessed", "idx2word.json")
    checkpoint_dir = os.path.join(_ROOT, "checkpoints")
    
    # 체크포인트 파일 찾기
    checkpoint_files = [
        os.path.join(checkpoint_dir, "best_model.pth"),
        os.path.join(checkpoint_dir, "checkpoint_epoch_20.pth"),
    ]
    
    checkpoint_path = None
    for cp in checkpoint_files:
        if os.path.exists(cp):
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        else:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_dir}")
    
    print(f"사용할 체크포인트: {checkpoint_path}")
    
    # 단어장 로드
    print("\n단어장 로드 중...")
    with open(word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)
    
    with open(idx2word_path, 'r', encoding='utf-8') as f:
        idx2word = json.load(f)
        idx2word = {int(k): v for k, v in idx2word.items()}
    
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
    
    # 테스트 데이터셋 생성
    print("\n테스트 데이터셋 생성 중...")
    test_dataset = Flickr8kImageCaptionDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        transform=transform,
        split="test"
    )
    print(f"테스트 데이터: {len(test_dataset)}개")
    
    # 모델 로드
    print("\n모델 로드 중...")
    model = load_model(checkpoint_path, vocab_size, device)
    
    # 시각화
    print("\n예측 캡션 생성 및 시각화 중...")
    visualize_predictions(
        model,
        test_dataset,
        idx2word,
        device,
        num_samples=5,
        start_token=word2idx.get('<start>', 1),
        end_token=word2idx.get('<end>', 2)
    )


if __name__ == '__main__':
    main()