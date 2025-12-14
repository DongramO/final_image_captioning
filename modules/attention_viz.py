"""
Attention 시각화 유틸 (heatmap overlay)

사용 예시:
    captions, attn_info = model.generate_caption(img_tensor, idx2word, return_attention=True)

    from attention_viz import save_attention_overlays

    save_attention_overlays(
        image=img_pil_or_tensor,              # PIL.Image 또는 torch.Tensor[C,H,W]
        words=attn_info[0]["words"],          # ["a","dog",...]
        alphas=attn_info[0]["alphas"],        # [Tensor[P], ...]
        spatial_hw=attn_info[0]["spatial_hw"],# (H,W) e.g., (7,7)
        out_dir="attn_out",
        prefix="sample",
    )

주의:
- spatial_hw가 None이면, alpha 길이로부터 sqrt 추정 시도합니다.
- 이미지가 normalize되어 있다면(예: ImageNet mean/std), denormalize를 직접 적용해 주세요.
"""

from __future__ import annotations

import os
import math
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def _to_pil(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()
        if x.dim() == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).contiguous()
        x = x.numpy()

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"지원하지 않는 image 타입: {type(image)}")


def _infer_hw_from_alpha(alpha_len: int) -> Tuple[int, int]:
    s = int(round(math.sqrt(alpha_len)))
    if s * s != alpha_len:
        w = max(s, 1)
        h = int(math.ceil(alpha_len / w))
        return h, w
    return s, s


def save_attention_overlays(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    words: List[str],
    alphas: List[torch.Tensor],
    spatial_hw: Optional[Tuple[int, int]] = None,
    out_dir: str = "attention_out",
    prefix: str = "attn",
    max_words: int = 30,
    overlay_alpha: float = 0.45,
    dpi: int = 160,
):
    """
    단어별 attention heatmap을 원본 이미지 위에 overlay해서 저장합니다.

    저장 파일:
      {out_dir}/{prefix}_{step:02d}_{word}.png

    Args:
        image: 원본 이미지 (PIL 또는 Tensor[C,H,W] 또는 ndarray[H,W,C])
        words: 캡션 단어 리스트
        alphas: 단어별 alpha 리스트, 각 alpha shape [P]
        spatial_hw: encoder feature map의 (H,W)
        out_dir: 저장 폴더
        prefix: 파일 prefix
        max_words: 최대 저장 단어 수
        overlay_alpha: heatmap 투명도
    """
    os.makedirs(out_dir, exist_ok=True)
    img = _to_pil(image)
    W_img, H_img = img.size

    n = min(len(words), len(alphas), max_words)

    for i in range(n):
        word = str(words[i]).replace("/", "_").replace("\\", "_").replace(" ", "_")
        alpha = alphas[i]
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().float().numpy()
        else:
            alpha = np.asarray(alpha, dtype=np.float32)

        P = int(alpha.shape[0])
        if spatial_hw is None:
            h_map, w_map = _infer_hw_from_alpha(P)
        else:
            h_map, w_map = spatial_hw

        attn = alpha.reshape(h_map, w_map)
        attn = attn - attn.min()
        if attn.max() > 1e-8:
            attn = attn / attn.max()

        attn_img = Image.fromarray((attn * 255).astype(np.uint8)).resize((W_img, H_img), resample=Image.BILINEAR)
        attn_arr = np.array(attn_img).astype(np.float32) / 255.0

        fig = plt.figure(figsize=(W_img / dpi, H_img / dpi), dpi=dpi)
        ax = plt.gca()
        ax.imshow(img)
        ax.imshow(attn_arr, alpha=overlay_alpha)
        ax.set_axis_off()
        ax.set_title(f"{i:02d}: {word}", fontsize=10)

        out_path = os.path.join(out_dir, f"{prefix}_{i:02d}_{word}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    return out_dir
