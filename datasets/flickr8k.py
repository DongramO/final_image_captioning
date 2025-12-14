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
import os, glob
from torch.utils.data import Dataset
import torch
import ast

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
      
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img)
        if normalize:
            arr = arr.astype(np.float32) / 255.0
    
        return img, arr
    

    def preprocess_image(self, image_size: Tuple[int, int] = (224, 224)):
    
        IMG_DIR = self.image_dir
        OUT_DIR = self.out_dir
        
        os.makedirs(OUT_DIR, exist_ok=True)

        TARGET_SIZE = image_size
        NORMALIZE = True

    
        files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
        print("이미지 개수:", len(files))

        sample_name = files[0]
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
        os.makedirs(self.caption_out_dir, exist_ok=True)

        with open(self.caption_file, "r") as f:
            for _ in range(5):
                print(f.readline().strip())
                
        # 텍스트 파일을 CSV -> pandas DataFrame으로 변환
        # 원본 파일(Flickr8k.token.txt)의 첫 줄이 헤더('image,caption')이므로 header=0으로 읽기
        df = pd.read_csv(self.caption_file, sep=",", header=0)
        
        # 컬럼명 확인 및 정리
        if 'image' not in df.columns or 'caption' not in df.columns:
            # 헤더가 없거나 다른 형식인 경우
            df = pd.read_csv(self.caption_file, sep=",", header=None)
            df.columns = ["image", "caption"]
            # 첫 줄이 헤더인 경우 제거
            if len(df) > 0 and str(df.iloc[0]['image']).lower() == 'image':
                df = df.iloc[1:].reset_index(drop=True)
                print(f"⚠️ 헤더 행 제거됨")
        
        
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
        max_len = 20 
        
        pad_idx = word2idx["<pad>"]

        df["caption_padded"] = df["caption_indices"].apply(
            lambda x: self.pad_sequence(x, max_len, pad_idx)
        )
        df[["caption_indices", "caption_padded"]].head()
        final_df = df[["image", "caption_padded"]]
        final_df.head()
        
        final_df.to_csv(os.path.join(self.caption_out_dir, "captions_padded.csv"), index=False)

        with open(os.path.join(self.caption_out_dir, "word2idx.json"), "w") as f:
            json.dump(word2idx, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.caption_out_dir, "idx2word.json"), "w") as f:
            json.dump(idx2word, f, ensure_ascii=False, indent=2)


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


class Flickr8kImageOnlyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # os.listdir 사용 (Windows에서 더 안정적)
        if not os.path.exists(image_dir):
            raise ValueError(f"이미지 디렉토리가 존재하지 않습니다: {image_dir}")
        
        all_files = os.listdir(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        self.image_paths = sorted([
            os.path.join(image_dir, f) 
            for f in all_files 
            if os.path.splitext(f)[1].lower() in {ext.lower() for ext in image_extensions}
        ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

class Flickr8kImageCaptionDataset(Dataset):
    """
    이미지와 캡션을 함께 반환하는 Flickr8k 데이터셋
    """
    
    def __init__(
        self, 
        image_dir: str,
        captions_file: str,
        transform=None,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            image_dir: 이미지 파일들이 있는 디렉토리 경로
            captions_file: 캡션이 저장된 CSV 파일 경로 (captions_padded.csv)
            transform: 이미지 전처리 변환 함수
            split: "train", "val", "test" 중 하나
            train_split: 학습 데이터 비율
            val_split: 검증 데이터 비율
            test_split: 테스트 데이터 비율
            seed: 랜덤 시드 (재현성을 위해)
        """
        self.image_dir = image_dir
        self.captions_file = captions_file
        self.transform = transform
        self.split = split
        
        # 분할 비율 검증
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "train_split + val_split + test_split must equal 1.0"
        
        # 캡션 데이터 로드
        print(f"캡션 파일 로드 중: {captions_file}")
        self.df = pd.read_csv(captions_file)
        print(f"총 캡션 개수: {len(self.df)}")
        
        # CSV 컬럼 확인
        print(f"CSV 컬럼: {list(self.df.columns)}")
        if 'image' not in self.df.columns:
            raise ValueError(f"CSV 파일에 'image' 컬럼이 없습니다. 컬럼: {list(self.df.columns)}")
        
        # image 컬럼의 샘플 값 확인
        print(f"image 컬럼 샘플 (처음 5개): {self.df['image'].head().tolist()}")
        
        # 빈 값이나 잘못된 값 확인
        empty_images = self.df[self.df['image'].isna() | (self.df['image'] == '') | (self.df['image'].str.strip() == '')]
        if len(empty_images) > 0:
            print(f"⚠️ 경고: 빈 image 값이 {len(empty_images)}개 있습니다.")
            print(f"빈 값 샘플: {empty_images.head()}")
        
        # 이미지 파일명으로 그룹화 (한 이미지에 여러 캡션이 있을 수 있음)
        self.image_groups = self.df.groupby('image')
        self.unique_images = sorted(self.image_groups.groups.keys())
        print(f"고유 이미지 개수: {len(self.unique_images)}")
        
        # unique_images 샘플 확인
        if len(self.unique_images) > 0:
            print(f"이미지 파일명 샘플 (처음 5개): {self.unique_images[:5]}")
        
        # 학습/검증/테스트 분할
        np.random.seed(seed)
        indices = np.random.permutation(len(self.unique_images))
        
        train_end = int(len(self.unique_images) * train_split)
        val_end = train_end + int(len(self.unique_images) * val_split)
        
        train_indices = set(indices[:train_end])
        val_indices = set(indices[train_end:val_end])
        test_indices = set(indices[val_end:])
        
        # 분할에 따라 이미지 선택
        if split == "train":
            self.selected_indices = [i for i in range(len(self.unique_images)) if i in train_indices]
        elif split == "val":
            self.selected_indices = [i for i in range(len(self.unique_images)) if i in val_indices]
        elif split == "test":
            self.selected_indices = [i for i in range(len(self.unique_images)) if i in test_indices]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        
        print(f"{split} split: {len(self.selected_indices)}개 이미지")
        
        # 선택된 이미지들의 모든 캡션을 리스트로 구성
        self.data_pairs = []
        for img_idx in self.selected_indices:
            image_name = self.unique_images[img_idx]
            image_group = self.image_groups.get_group(image_name)
            
            for _, row in image_group.iterrows():
                caption_str = row['caption_padded']
                
                # 문자열을 리스트로 파싱
                try:
                    caption_list = ast.literal_eval(caption_str)
                except:
                    caption_list = eval(caption_str)
                
                self.data_pairs.append({
                    'image_name': image_name,
                    'caption': caption_list
                })
        
        print(f"{split} split: 총 {len(self.data_pairs)}개 (이미지, 캡션) 쌍")
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: 데이터 인덱스
            
        Returns:
            image: 전처리된 이미지 텐서 [C, H, W]
            caption: 캡션 텐서 [seq_length] (LongTensor)
        """
        data_pair = self.data_pairs[idx]
        image_name = data_pair['image_name']
        caption = data_pair['caption']
        
        # image_name 검증
        if not image_name or image_name.strip() == '':
            raise ValueError(f"idx {idx}: image_name이 비어있습니다. data_pair: {data_pair}")
        
        # 이미지 로드
        image_path = os.path.join(self.image_dir, image_name)
        
        if not os.path.exists(image_path):
            # 디버깅 정보 출력
            print(f"⚠️ 이미지 파일을 찾을 수 없습니다:")
            print(f"  - idx: {idx}")
            print(f"  - image_name: '{image_name}'")
            print(f"  - image_dir: {self.image_dir}")
            print(f"  - image_path: {image_path}")
            print(f"  - image_dir 존재 여부: {os.path.exists(self.image_dir)}")
            if os.path.exists(self.image_dir):
                # 디렉토리 내 파일 샘플 확인
                files = os.listdir(self.image_dir)
                print(f"  - image_dir 내 파일 개수: {len(files)}")
                if len(files) > 0:
                    print(f"  - 파일 샘플 (처음 5개): {files[:5]}")
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Transform 적용
        if self.transform:
            image = self.transform(image)
        
        # 캡션을 텐서로 변환
        caption_tensor = torch.tensor(caption, dtype=torch.long)
        
        return image, caption_tensor
    
    def get_image_name(self, idx):
        """인덱스에 해당하는 이미지 파일명 반환"""
        return self.data_pairs[idx]['image_name']