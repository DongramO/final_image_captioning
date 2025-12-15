"""
이미지 캡셔닝 통합 모델

변경점(Attention alpha 저장/시각화 준비):
- encoder 출력이 (global, spatial, (H,W)) 형태면 spatial을 decoder에 encoder_out으로 전달
- generate_caption()에서 return_attention=True일 때, 각 토큰별 alpha를 함께 반환
"""

import torch
import torch.nn as nn


class ImageCaptionModel(nn.Module):
    """
    Encoder-Decoder 구조의 이미지 캡셔닝 모델
    """

    def __init__(self, encoder, decoder, vocab_size, embed_size=256, hidden_size=512):
        super(ImageCaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def forward(self, images, captions=None):
        # 1) 이미지 인코딩
        encoded = self.encoder(images)

        # encoder가 attention용 spatial을 같이 주는 경우
        encoder_out = None
        if isinstance(encoded, (tuple, list)) and len(encoded) >= 2:
            features = encoded[0]
            encoder_out = encoded[1]
        else:
            features = encoded

        # 2) 디코딩(Teacher Forcing)
        if captions is None:
            raise ValueError("학습 시에는 captions가 필요합니다.")
        outputs = self.decoder(features, captions, encoder_out=encoder_out)  # [B, T, vocab]
        return outputs

    def generate_caption(
        self,
        images,
        idx2word,
        max_length=50,
        start_token=1,
        end_token=2,
        beam_size=1,
        show_topk=True,
        temperature=1.0,
        sample=False,          # True면 sampling(multinomial), False면 greedy(argmax)
        topk=10,
        return_attention=False,
        repetition_penalty=1.2,  # 반복 억제 강도 (1.0 = 없음, >1.0 = 억제)
        no_repeat_ngram_size=3,   # N-gram 반복 방지 (최근 N개 단어)
        use_topk_sampling=False,  # Top-k sampling 사용 여부
    ):
        """
        추론 시 사용: 이미지로부터 캡션 생성

        return_attention=True이면:
        - caption_strings: List[str]
        - attn_info: List[dict]
            {
              "words": List[str],            # 최종 캡션에 포함된 단어들
              "alphas": List[Tensor],        # 각 단어별 alpha, shape [P]
              "spatial_hw": (H, W) or None,  # encoder layer4 map 크기
            }
        """
        self.eval()
        with torch.no_grad():
            if images.dim() == 3:
                images = images.unsqueeze(0)  # [1, C, H, W]

            device = images.device
            encoded = self.encoder(images)

            encoder_out = None
            spatial_hw = None
            if isinstance(encoded, (tuple, list)) and len(encoded) >= 2:
                features = encoded[0]
                encoder_out = encoded[1]
                print(f"[DEBUG] encoder_out shape: {encoder_out.shape if encoder_out is not None else None}")
                if len(encoded) >= 3:
                    spatial_hw = encoded[2]
            else:
                features = encoded
                print("[⚠️ 경고] encoder_out이 None입니다! Attention이 작동하지 않습니다!")

            if beam_size != 1:
                # beam은 미구현: 필요하면 붙이는게 맞고, 지금은 greedy/sampling으로 통일
                pass

            captions, attn_raw = self._greedy_search(
                features=features,
                encoder_out=encoder_out,
                max_length=max_length,
                start_token=start_token,
                end_token=end_token,
                device=device,
                idx2word=idx2word,
                show_topk=show_topk,
                topk=topk,
                temperature=temperature,
                sample=sample,
                return_attention=return_attention,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                use_topk_sampling=use_topk_sampling,
            )

            caption_strings = []
            attn_info = []

            for b, caption in enumerate(captions):
                if isinstance(caption, torch.Tensor):
                    caption = caption.cpu().tolist()

                words = []
                alphas = []

                # attn_raw[b]는 step별 alpha 리스트
                raw_alphas = attn_raw[b] if (return_attention and attn_raw is not None) else None

                # caption = [<start>, w1, w2, ..., <end>, ...]
                # raw_alphas는 w1, w2, ... 생성 시점의 alpha에 대응 (len ~= caption_len-1)
                alpha_step = 0
                for idx in caption[1:]:  # start 제외
                    idx = int(idx)
                    if idx == end_token:
                        break

                    word = idx2word.get(idx, "<unk>")

                    # special token은 캡션 표시에서 제외
                    if word in ["<pad>", "<start>", "<end>"]:
                        alpha_step += 1
                        continue

                    if return_attention and raw_alphas is not None and alpha_step < len(raw_alphas):
                        alphas.append(raw_alphas[alpha_step].detach().cpu())  # [P]
                    alpha_step += 1
                    words.append(word)

                caption_strings.append(" ".join(words) if words else "<unk>")

                if return_attention:
                    attn_info.append({
                        "words": words,
                        "alphas": alphas,
                        "spatial_hw": spatial_hw,
                    })

            if return_attention:
                return caption_strings, attn_info
            return caption_strings

    def _greedy_search(
        self,
        features,
        encoder_out,
        max_length,
        start_token,
        end_token,
        device,
        idx2word=None,
        show_topk=False,
        topk=10,
        temperature=1.0,
        sample=False,
        return_attention=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        use_topk_sampling=False,
    ):
        """
        Greedy / Sampling 공용 디코딩
        - sample=False: greedy(argmax)
        - sample=True : multinomial sampling
        - use_topk_sampling=True: top-k 중에서만 선택
        """
        batch_size = features.size(0)
        print(f"[DEBUG] temperature={temperature}, sample={sample}, topk={topk}, repetition_penalty={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size}")

        # 초기 상태 (spatial feature도 함께 전달)
        h_states, c_states = self.decoder.init_hidden_state(features, encoder_out=encoder_out)

        captions = [[start_token] for _ in range(batch_size)]
        attn_per_sample = [[] for _ in range(batch_size)] if return_attention else None

        for b in range(batch_size):
            h = [hs[b:b+1].contiguous() for hs in h_states]  # [1, H]
            c = [cs[b:b+1].contiguous() for cs in c_states]  # [1, H]

            enc_b = encoder_out[b:b+1] if encoder_out is not None else None

            current = torch.tensor([start_token], dtype=torch.long, device=device)  # [1]

            for step in range(max_length):
                word_embed = self.decoder.embedding(current)  # [1, E]

                logits, h, c, alpha = self.decoder.step(
                    word_embed=word_embed,
                    h_states=h,
                    c_states=c,
                    encoder_out=enc_b,
                    return_alpha=return_attention,
                )  # logits: [1, V], alpha: [1, P] or None

                # 강화된 반복 억제 메커니즘
                if len(captions[b]) >= 2:
                    # 1. 최근 N개 단어에 대한 반복 억제 (N-gram 반복 방지)
                    if no_repeat_ngram_size > 0 and len(captions[b]) >= no_repeat_ngram_size:
                        # 최근 N개 단어를 확인
                        recent_tokens = captions[b][-(no_repeat_ngram_size-1):]
                        for token in recent_tokens:
                            if token != start_token and token != end_token:
                                # 반복된 토큰의 logit을 낮춤
                                logits[0, token] = logits[0, token] / repetition_penalty
                    
                    # 2. 직전 토큰 강력 억제
                    last_token = captions[b][-1]
                    if last_token != start_token and last_token != end_token:
                        logits[0, last_token] = logits[0, last_token] / (repetition_penalty * 2)

                if temperature is None or temperature <= 0:
                    temperature = 1.0
                logits = logits / temperature

                # 디버깅 Top-k (첫 샘플만 Step 0~3)
                if show_topk and idx2word is not None and b == 0 and step <= 3:
                    self._print_topk(logits.squeeze(0), idx2word, k=topk, step=step)

                # Top-k sampling 또는 일반 sampling/greedy
                if use_topk_sampling:
                    # Top-k 중에서만 선택
                    topk_logits, topk_indices = torch.topk(logits, min(topk, logits.size(-1)), dim=-1)
                    topk_probs = torch.softmax(topk_logits, dim=-1)
                    
                    if sample:
                        # Top-k 중에서 multinomial sampling
                        sampled_idx = torch.multinomial(topk_probs, num_samples=1)
                        predicted = topk_indices.gather(1, sampled_idx).view(-1)
                    else:
                        # Top-k 중에서 가장 높은 것 선택 (greedy)
                        predicted = topk_indices[0, 0:1]
                elif sample:
                    probs = torch.softmax(logits, dim=-1)  # [1, V]
                    predicted = torch.multinomial(probs, num_samples=1).view(-1)  # [1]
                else:
                    predicted = torch.argmax(logits, dim=-1)  # [1]

                pred_idx = int(predicted.item())
                captions[b].append(pred_idx)

                if return_attention and alpha is not None:
                    # alpha[0]: [P]
                    attn_per_sample[b].append(alpha[0].detach().cpu())

                if pred_idx == end_token:
                    break

                current = predicted  # [1]

        # 텐서화 (길이 맞추기: 남는 건 end_token으로 패딩)
        max_len = max(len(seq) for seq in captions)
        out = []
        for seq in captions:
            seq = seq + [end_token] * (max_len - len(seq))
            out.append(seq[:max_len])

        return torch.tensor(out, dtype=torch.long, device=device), attn_per_sample

    def _print_topk(self, logits, idx2word, k=10, step=0):
        """
        logits: [V]
        """
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

        print(f"\n[Step {step}] Top-{k} 예측:")
        for i in range(k):
            idx = int(topk_indices[i].item())
            prob = float(topk_probs[i].item())
            word = idx2word.get(idx, "<unk>")
            print(f"  {i+1}. {word:15s} (idx: {idx:5d}, prob: {prob:.4f})")
