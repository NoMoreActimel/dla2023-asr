import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # input fields: ["audio", "spectrogram", "duration","text", "text_encoded", "audio_path"]
    # output fields: ["spectrogram", "text_encoded", "text_encoded_length", "text"]
    return {
        'spectrogram': torch.stack([row['spectrogram'] for row in dataset_items], axis=0),
        'text_encoded': torch.stack([row['text_encoded'] for row in dataset_items], axis=0),
        'text_encoded_length': torch.tensor([len(row['text']) for row in dataset_items], dtype=torch.int32),
        'text': [row['text'] for row in dataset_items]
    }