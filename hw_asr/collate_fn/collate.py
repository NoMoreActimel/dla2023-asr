import logging
import torch

from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # input fields: ["audio", "spectrogram", "duration","text", "text_encoded", "audio_path"]
    # output fields: ["spectrogram", "text_encoded", "text_encoded_length", "text"]
    return {
        'spectrogram': pad_sequence([row['spectrogram'].squeeze(0).T for row in dataset_items], batch_first=True),
        'text_encoded': pad_sequence([row['text_encoded'].T for row in dataset_items], batch_first=True),
        'text_encoded_length': torch.tensor([len(row['text']) for row in dataset_items], dtype=torch.int32),
        'text': [row['text'] for row in dataset_items]
    }