"""
이미지 인코더 모듈
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def encode_images(encoder, dataloader, device="cuda"):
    encoder = encoder.to(device)
    encoder.eval()

    all_paths = []
    all_embs = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device, non_blocking=True)
            embs = encoder(images)          # [B, embed_size]
            embs = embs.cpu()

            all_embs.append(embs)
            all_paths.extend(list(paths))

    all_embs = torch.cat(all_embs, dim=0)   # [N, embed_size]
    return all_paths, all_embs