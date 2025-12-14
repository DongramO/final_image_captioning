"""
이미지 캡셔닝 통합 모델
"""

import torch
import torch.nn as nn
from modules.decoder import CaptionDecoder


class ImageCaptionModel(nn.Module):
    """
    Encoder-Decoder 구조의 이미지 캡셔닝 모델
    """
    
    def __init__(self, encoder, decoder, vocab_size, embed_size=256, hidden_size=512):
        """
        Args:
            encoder: 이미지 인코더 모델 (예: ResNet)
            decoder: 캡션 디코더 모델 (CaptionDecoder)
            vocab_size: 단어장 크기
            embed_size: 임베딩 차원 (이미지 특징 차원)
            hidden_size: 히든 상태 차원
        """
        super(ImageCaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
    def forward(self, images, captions=None):
        """
        학습 시 사용: 이미지와 캡션을 받아서 다음 단어를 예측
        
        Args:
            images: 입력 이미지 텐서 [batch_size, channels, height, width]
            captions: 캡션 텐서 [batch_size, seq_length] (<start> 토큰 포함, <end> 토큰 제외)
            
        Returns:
            outputs: 생성된 캡션 로짓 [batch_size, seq_length-1, vocab_size]
        """
        # 1. 이미지 인코딩
        features = self.encoder(images)  # [batch_size, embed_size]
        
        # 2. 캡션 디코딩 (Teacher Forcing)
        if captions is None:
            raise ValueError("학습 시에는 captions가 필요합니다.")
        
        outputs = self.decoder(features, captions)  # [batch_size, seq_length-1, vocab_size]
        
        return outputs
    
    def generate_caption(self, images, idx2word, max_length=50, start_token=1, end_token=2, beam_size=1):
        """
        추론 시 사용: 이미지로부터 캡션 생성
        
        Args:
            images: 입력 이미지 텐서 [batch_size, channels, height, width] 또는 [1, channels, height, width]
            idx2word: 인덱스를 단어로 변환하는 딕셔너리
            max_length: 최대 생성 길이
            start_token: 시작 토큰 인덱스
            end_token: 종료 토큰 인덱스
            beam_size: Beam search 크기 (1이면 Greedy search)
            
        Returns:
            captions: 생성된 캡션 리스트 (문자열 리스트)
        """
        self.eval()
        
        with torch.no_grad():
            # 배치 차원 확인
            if images.dim() == 3:
                images = images.unsqueeze(0)  # [1, C, H, W]
            
            batch_size = images.size(0)
            device = images.device
            
            # 1. 이미지 인코딩
            features = self.encoder(images)  # [batch_size, embed_size]
            
            # 2. Greedy search 또는 Beam search로 캡션 생성
            if beam_size == 1:
                captions = self._greedy_search(
                    features, max_length, start_token, end_token, device
                )
            else:
                captions = self._beam_search(
                    features, max_length, start_token, end_token, beam_size, device
                )
            
            # 3. 인덱스를 단어로 변환
            caption_strings = []
            for caption in captions:
                words = []
                # 텐서를 리스트로 변환
                if isinstance(caption, torch.Tensor):
                    caption = caption.cpu().tolist()
                
                for idx in caption:
                    if idx == end_token:
                        break
                    if idx == start_token:
                        continue
                    
                    # 인덱스 타입 확인 및 변환
                    idx = int(idx)
                    
                    # 디버깅: 예측된 인덱스 확인
                    if len(words) == 0:  # 첫 번째 예측만 출력
                        print(f"  [디버깅] 예측된 인덱스: {idx}, vocab_size: {self.vocab_size}")
                        print(f"  [디버깅] idx2word에 있는지: {idx in idx2word}")
                        if idx not in idx2word and idx < len(idx2word):
                            print(f"  [디버깅] idx2word 키 타입: {type(list(idx2word.keys())[0])}")
                    
                    if idx in idx2word:
                        word = idx2word[idx]
                        if word not in ['<pad>', '<start>', '<end>', '<unk>']:
                            words.append(word)
                    else:
                        # 인덱스가 범위를 벗어난 경우
                        if idx >= self.vocab_size:
                            print(f"  [경고] 인덱스 {idx}가 vocab_size {self.vocab_size}를 초과합니다!")
                        words.append('<unk>')
                
                caption_strings.append(' '.join(words) if words else '<unk>')
            
            return caption_strings
    
    def _greedy_search(self, features, max_length, start_token, end_token, device):
        """
        Greedy search로 캡션 생성
        
        Args:
            features: 이미지 특징 [batch_size, embed_size]
            max_length: 최대 생성 길이
            start_token: 시작 토큰 인덱스
            end_token: 종료 토큰 인덱스
            device: 디바이스
            
        Returns:
            captions: 생성된 캡션 인덱스 리스트 [batch_size, seq_length]
        """
        batch_size = features.size(0)
        
        # 초기 상태 설정
        h_states, c_states = self.decoder.init_hidden_state(features)
        
        # 시작 토큰으로 초기화
        inputs = torch.full((batch_size,), start_token, dtype=torch.long, device=device)
        
        captions = []
        for _ in range(batch_size):
            captions.append([start_token])
        
        # 각 샘플별로 독립적으로 생성
        for batch_idx in range(batch_size):
            h_batch = [h[batch_idx:batch_idx+1] for h in h_states]  # [1, hidden_size]
            c_batch = [c[batch_idx:batch_idx+1] for c in c_states]  # [1, hidden_size]
            feat_batch = features[batch_idx:batch_idx+1]  # [1, embed_size]
            
            current_input = torch.tensor([start_token], dtype=torch.long, device=device)
            
            for _ in range(max_length):
                # 현재 입력 임베딩
                x_t = self.decoder.embedding(current_input)  # [1, embed_size]
                
                # 각 레이어를 순차적으로 통과
                for layer_idx in range(self.decoder.num_layers):
                    h_batch[layer_idx], c_batch[layer_idx] = self.decoder.lstm_cells[layer_idx](
                        x_t, h_batch[layer_idx], c_batch[layer_idx]
                    )
                    x_t = h_batch[layer_idx]
                
                # 출력 로짓
                outputs = self.decoder.linear(h_batch[-1])  # [1, vocab_size]
                
                # Greedy: 가장 높은 확률의 단어 선택
                predicted = outputs.argmax(1)  # [1]
                predicted_idx = predicted.item()
                
                captions[batch_idx].append(predicted_idx)
                
                # <end> 토큰이 나오면 종료
                if predicted_idx == end_token:
                    break
                
                # 다음 입력으로 사용
                current_input = predicted
        
        # 텐서로 변환
        max_len = max(len(c) for c in captions)
        caption_tensors = []
        for caption in captions:
            padded = caption + [end_token] * (max_len - len(caption))
            caption_tensors.append(padded[:max_len])
        
        return torch.tensor(caption_tensors, dtype=torch.long, device=device)
    
    def _beam_search(self, features, max_length, start_token, end_token, beam_size, device):
        """
        Beam search로 캡션 생성 (간단한 구현)
        
        Args:
            features: 이미지 특징 [batch_size, embed_size]
            max_length: 최대 생성 길이
            start_token: 시작 토큰 인덱스
            end_token: 종료 토큰 인덱스
            beam_size: Beam search 크기
            device: 디바이스
            
        Returns:
            captions: 생성된 캡션 인덱스 리스트 [batch_size, seq_length]
        """
        # 간단한 구현: 현재는 Greedy search와 동일하게 처리
        # 나중에 더 정교한 Beam search 구현 가능
        return self._greedy_search(features, max_length, start_token, end_token, device)

