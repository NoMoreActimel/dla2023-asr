from typing import List

import numpy as np

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamsearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        assert hasattr(self.text_encoder, "ctc_beam_search"), "Incompatible text encoder, ctc_beam_search is needed"

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []

        predicted_texts = np.array([
            self.text_encoder.ctc_beam_search(torch.exp(log_prob), length)[0].text
            for log_prob, length in zip(log_probs, log_probs_length)
        ])
        lengths = log_probs_length.detach().numpy()
        
        for pred_text, length, target_text in zip(predicted_texts, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))

        return sum(cers) / len(cers)
