"""
이미지 인코더 모듈

변경점:
- encoder가 (global, spatial, ...) 형태를 반환할 수 있으므로 encode_images에서는 global만 취급
"""

import torch

def encode_images(encoder, dataloader, device="cuda"):
    encoder = encoder.to(device)
    encoder.eval()

    all_paths = []
    all_embs = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device, non_blocking=True)
            embs = encoder(images)

            # attention 확장 반환 처리
            if isinstance(embs, (tuple, list)):
                embs = embs[0]

            embs = embs.cpu()
            all_embs.append(embs)
            all_paths.extend(list(paths))

    all_embs = torch.cat(all_embs, dim=0)
    return all_paths, all_embs
