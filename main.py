import modules
import torch
import os
import torch.nn as nn
from torchvision import transforms
import datasets
from torch.utils.data import DataLoader
from modules.encoder import encode_images

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 너가 만든 직접 구현 Encoder 사용 (예: ResNet18Encoder(embed_size=256))
    encoder = modules.ResNet(embed_size=256)
    
    _ROOT = os.path.dirname(os.path.dirname(__file__))

    dataset = datasets.Flickr8kImageOnlyDataset(
        image_dir=os.path.join(_ROOT, "final_image_captioning", "datasets", "data", "Flickr8k_images"),
        transform=transform
    )
    
    print('***', len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    paths, embs = encode_images(encoder, loader, device=device)
    print(len(paths), embs.shape)  # (N, torch.Size([N, 256]))
    print(embs.mean(), embs.std())
