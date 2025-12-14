import os
import torch
import torch.nn as nn
from torchvision import transforms
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
    # 텐서를 numpy로 변환
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
    테스트 셋의 이미지와 예측 캡션을 시각화
    
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
    import random
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 이미지와 실제 캡션 로드
            image, true_caption = test_dataset[idx]
            image_name = test_dataset.get_image_name(idx)
            
            # 이미지를 배치로 변환
            image_batch = image.unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 캡션 생성
            predicted_captions = model.generate_caption(
                image_batch,
                idx2word,
                max_length=50,
                start_token=start_token,
                end_token=end_token,
                beam_size=1
            )
            predicted_caption = predicted_captions[0]

            # 디버깅: 예측 과정 확인
            print(f"\n[디버깅] 샘플 {i+1}:")
            print(f"  vocab_size: {len(idx2word)}")
            print(f"  idx2word 키 개수: {len(idx2word)}")
            print(f"  예측 캡션 (원본): {predicted_caption}")

            # _greedy_search의 반환값 직접 확인
            with torch.no_grad():
                features = model.encoder(image_batch)
                captions_tensor = model._greedy_search(
                    features, 50, start_token, end_token, device
                )
                print(f"  예측된 인덱스 (텐서): {captions_tensor[0][:10].cpu().tolist()}")
                print(f"  예측된 인덱스 범위: {captions_tensor[0].min().item()} ~ {captions_tensor[0].max().item()}")

            # 디버깅: 모델의 출력 확인
            with torch.no_grad():
                features = model.encoder(image_batch)
                h_states, c_states = model.decoder.init_hidden_state(features)
                
                # 첫 번째 단어 예측 확인
                start_input = torch.tensor([start_token], dtype=torch.long, device=device)  # 수정
                x_t = model.decoder.embedding(start_input)  # [1, embed_size]
                
                for layer_idx in range(model.decoder.num_layers):
                    h_states[layer_idx], c_states[layer_idx] = model.decoder.lstm_cells[layer_idx](
                        x_t, h_states[layer_idx], c_states[layer_idx]
                    )
                    x_t = h_states[layer_idx]
                
                outputs = model.decoder.linear(h_states[-1])
                predicted_idx = outputs.argmax(1).item()
                
                print(f"\n[디버깅] 첫 번째 예측:")
                print(f"  예측 인덱스: {predicted_idx}")
                print(f"  vocab_size: {model.vocab_size}")
                print(f"  출력 로짓 shape: {outputs.shape}")
                print(f"  출력 로짓 최대값: {outputs.max().item():.4f}")
                print(f"  출력 로짓 평균: {outputs.mean().item():.4f}")
                print(f"  idx2word에 있는지: {predicted_idx in idx2word}")
                if predicted_idx in idx2word:
                    print(f"  예측 단어: {idx2word[predicted_idx]}")
                else:
                    print(f"  ⚠ 인덱스 {predicted_idx}가 idx2word에 없습니다!")
                    print(f"  idx2word 키 범위: {min(idx2word.keys())} ~ {max(idx2word.keys())}")
            
            # 실제 캡션 변환 (인덱스 -> 단어)
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
            
            # 이미지 denormalize
            img_display = denormalize_image(image)
            
            # 시각화
            axes[i].imshow(img_display)
            axes[i].axis('off')
            axes[i].set_title(
                f"이미지: {image_name}\n"
                f"실제 캡션: {true_caption_str}\n"
                f"예측 캡션: {predicted_caption}",
                fontsize=10
            )
    
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
    print("시각화 저장 완료: test_predictions.png")
    plt.show()


def load_model(checkpoint_path, vocab_size, device):
    """
    체크포인트에서 모델 로드 (설정도 함께 읽기)
    """
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 체크포인트에서 설정 읽기
    embed_size = 128   # 256 → 128로 변경
    hidden_size = 256  # 512 → 256로 변경
    num_layers = 1 
    
    print(f"체크포인트에서 읽은 설정:")
    print(f"  embed_size: {embed_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  vocab_size: {vocab_size}")
    
    # 모델 생성 (체크포인트의 설정 사용)
    encoder = modules.ResNet(embed_size=embed_size)
    decoder = CaptionDecoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=0.5
    )
    
    model = ImageCaptionModel(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size
    )
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"모델 로드 완료: {checkpoint_path}")
    print(f"  에폭: {checkpoint['epoch']}, 손실: {checkpoint['loss']:.4f}")
    
    return model


def main():
    # 설정
    _ROOT = os.path.dirname(os.path.dirname(__file__))
    
    # 모델 설정 (학습 시 사용한 설정과 동일해야 함)
    embed_size = 256  # 학습 시 사용한 값
    hidden_size = 512
    num_layers = 2
    dropout = 0.5
    
    # 경로 설정
    image_dir = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "Flickr8k_images")
    captions_file = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "captions_preprocessed", "captions_padded.csv")
    word2idx_path = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "captions_preprocessed", "word2idx.json")
    idx2word_path = os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "captions_preprocessed", "idx2word.json")
    checkpoint_dir = os.path.join(_ROOT, "checkpoints")
    
    # 체크포인트 파일 선택 (최신 또는 최고 성능 모델)
    checkpoint_files = [
        os.path.join(checkpoint_dir, "best_model.pth"),
        os.path.join(checkpoint_dir, "checkpoint_epoch_20.pth"),
        # 다른 체크포인트 파일들...
    ]
    
    # 존재하는 체크포인트 찾기
    checkpoint_path = None
    for cp in checkpoint_files:
        if os.path.exists(cp):
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        # 체크포인트 디렉토리에서 가장 최근 파일 찾기
        import glob
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
        # JSON에서 키가 문자열로 저장되어 있을 수 있으므로 정수로 변환
        idx2word = {int(k): v for k, v in idx2word.items()}
    
    vocab_size = len(word2idx)
    print(f"단어장 크기: {vocab_size}")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Transform 설정 (학습 시와 동일)
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
        num_samples=10,  # 5개 샘플 시각화
        start_token=word2idx.get('<start>', 1),
        end_token=word2idx.get('<end>', 2)
    )


if __name__ == '__main__':
    main()