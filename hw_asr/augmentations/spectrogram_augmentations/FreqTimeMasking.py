import torchaudio
from torch import Tensor
from torch import nn

from hw_asr.augmentations.base import AugmentationBase

class FreqTimeMaskingSpecAug(AugmentationBase):
    def __init__(self, max_freq_mask=20, max_time_mask=100, *args, **kwargs):
        self._aug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(max_freq_mask),
            torchaudio.transforms.TimeMasking(max_time_mask),
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
