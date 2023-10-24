import numpy as np
import torchaudio
from torch import Tensor
from torch import nn

from hw_asr.augmentations.base import AugmentationBase

class TimeStretchSpecAug(AugmentationBase):
    def __init__(self, stretch_min=0.8, stretch_max=1.2, n_stretcn=3, n_freq=128, *args, **kwargs):
        self.stretch_rates = np.linspace(stretch_min, stretch_max, n_stretcn)
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        stretch_index = np.random.choice(self.stretch_rates.shape[0])
        return self._aug(x, self.stretch_rates[stretch_index]).squeeze(1)
