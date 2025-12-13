"""
Flickr8k 데이터셋 로더
"""

import os
import numpy as np  
import pandas as pd
from PIL import Image
from typing import List, Tuple
import re
from collections import Counter
import json

_ROOT = os.path.dirname(os.path.dirname(__file__))
class Flickr8kDataset():
    
    def __init__(self, transform=None, vocab=None):
        base = os.path.join(_ROOT, "datasets", "data")
        self.image_dir = os.path.join(base, "Flickr8k_images")
        self.out_dir = os.path.join(base, "Flickr8k_preprocessed")
        self.caption_file = os.path.join(base, "Flickr8k.token.txt")
        self.caption_out_dir = os.path.join(base, "captions_preprocessed")
        self.dataset_type = "flickr8k"
        self.transform = transform
        self.vocab = vocab
        
    
    def preprocess_one_image(self, image_path, target_size=(224,224), normalize=True):
        # 1) 이미지 열기 + RGB로 통일
        img = Image.open(image_path).convert("RGB")
        
        # 2) 리사이즈
        img = img.resize(target_size)
        
        # 3) numpy 배열로 변환
        arr = np.array(img)  # shape: (224,224,3), 값: 0~255
        
        # 4) 정규화(0~1)
        if normalize:
            arr = arr.astype(np.float32) / 255.0
    
        return img, arr
    

    def preprocess_image(self, image_size: Tuple[int, int] = (224, 224)):
    
        IMG_DIR = self.image_dir
        OUT_DIR = self.out_dir
        
        os.makedirs(OUT_DIR, exist_ok=True)

        TARGET_SIZE = image_size
        NORMALIZE = True           # True면 0~1로 나눔(모델 학습에 자주 사용)

    
        files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
        print("이미지 개수:", len(files))

        sample_name = files[0]
        print(sample_name)
        sample_path = os.path.join(IMG_DIR, sample_name)

        img, arr = self.preprocess_one_image(sample_path, TARGET_SIZE, NORMALIZE)

        print("샘플 파일:", sample_name)
        print("PIL 이미지 크기:", img.size)
        print("numpy shape:", arr.shape)
        print("min/max:", arr.min(), arr.max())
        
        
        fail_list = []
        done = 0

        for i, fname in enumerate(files):
            in_path = os.path.join(IMG_DIR, fname)
            out_path = os.path.join(OUT_DIR, fname)  # 같은 이름으로 저장
            
            try:
                img = Image.open(in_path).convert("RGB")
                img = img.resize(TARGET_SIZE)
                img.save(out_path, format="JPEG", quality=95)
                done += 1
                
                # 진행상황 출력(너무 많이 나오면 느려서 500개마다)
                if (i+1) % 500 == 0:
                    print(f"{i+1}/{len(files)} 처리 중... (성공 {done}, 실패 {len(fail_list)})")
                    
            except Exception as e:
                fail_list.append((fname, str(e)))

        print("✅ 완료!")
        print("성공:", done)
        print("실패:", len(fail_list))

        # 실패한 파일 일부 보기
        print("실패 예시 5개:", fail_list[:5])
        
        
    def load_captions_to_df(self):
        with open(self.caption_file, "r") as f:
            for _ in range(5):
                print(f.readline().strip())
                
        # 텍스트 파일을 CSV -> pandas DataFrame으로 변환
        df = pd.read_csv(self.caption_file, sep=",", header=None)
        df.columns = ["image", "caption"]
        
        
        df["caption"] = df["caption"].astype(str).str.lower().str.strip()
        print(df.head())
        
        df["caption_clean"] = df["caption"].apply(self.clean_text)
        print(df[["caption", "caption_clean"]].head())
        
        df["tokens"] = df["caption_clean"].apply(lambda x: x.split())
        
        out = df[["image", "caption_clean", "tokens"]]
        
        out.to_csv(self.caption_out_dir + "/captions_preprocessed.csv", index=False)
        print("저장 완료: captions_preprocessed.csv")
        
        all_words = []
        for tokens in df["tokens"]:
            all_words.extend(tokens)

        # 2. 단어 빈도 세기
        word_counts = Counter(all_words)

        # 3. 특수 토큰 정의
        special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]

        # 4. 단어 → 숫자 매핑 만들기
        word2idx = {}
        idx2word = {}

        idx = 0
        for token in special_tokens:
            word2idx[token] = idx
            idx2word[idx] = token
            idx += 1

        for word in word_counts:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1

        print("단어 개수:", len(word2idx))
        
        
        df["caption_indices"] = df["tokens"].apply(lambda x: self.tokens_to_indices(x, word2idx))

        # 결과 확인
        print(df[["tokens", "caption_indices"]].head())    
        df["length"] = df["caption_indices"].apply(len)

        max_len = df["length"].max()
        print("최대 문장 길이:", max_len)        
        max_len = 30 
        
        
        pad_idx = word2idx["<pad>"]

        df["caption_padded"] = df["caption_indices"].apply(
            lambda x: self.pad_sequence(x, max_len, pad_idx)
        )
        df[["caption_indices", "caption_padded"]].head()
        final_df = df[["image", "caption_padded"]]
        final_df.head()
        
        final_df.to_csv(self.caption_out_dir + "/captions_padded.csv", index=False)
        with open(self.caption_out_dir+"/word2idx.json", "w") as f:
            json.dump(word2idx, f, ensure_ascii=False, indent=2)

        with open(self.caption_out_dir+"/idx2word.json", "w") as f:
            json.dump(idx2word, f, ensure_ascii=False, indent=2)

        return final_df

    @staticmethod
    def pad_sequence(seq, max_len, pad_idx):
        if len(seq) < max_len:
            return seq + [pad_idx] * (max_len - len(seq))
        else:
            return seq[:max_len]
        
    @staticmethod
    def clean_text(s: str) -> str:
        s = re.sub(r"[^a-z ]", " ", s)   # 영어/공백만 남기고 나머지는 공백 처리
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    @staticmethod
    def tokens_to_indices(tokens, word2idx):
        indices = []
        
        # <start> 토큰 추가
        indices.append(word2idx["<start>"])
        
        for token in tokens:
            if token in word2idx:
                indices.append(word2idx[token])
            else:
                indices.append(word2idx["<unk>"])
        
        # <end> 토큰 추가
        indices.append(word2idx["<end>"])
        
        return indices