"""
데이터셋 로더 모듈
"""

from .flickr8k import Flickr8kDataset
from .coco import CocoDataset

__all__ = ['Flickr8kDataset', 'CocoDataset']

