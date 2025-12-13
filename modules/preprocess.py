import torch
import os
from PIL import Image
from typing import List, Tuple
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.flickr8k import Flickr8kDataset

def preprocess_datasets(image_size: Tuple[int, int] = (224, 224), tokenizer=None, vocab=None):
    
    dataset = Flickr8kDataset() 

    if dataset.dataset_type == "flickr8k":
        preprocess_image(dataset, image_size)
        preprocess_text(dataset, tokenizer, vocab)


def preprocess_image(dataobject, image_size: Tuple[int, int] = (224, 224)):
   dataobject.preprocess_image(image_size)
   
def preprocess_text(dataobject, tokenizer=None, vocab=None) -> List[int]:
    
    dataobject.load_captions_to_df()


# preprocess_datasets()