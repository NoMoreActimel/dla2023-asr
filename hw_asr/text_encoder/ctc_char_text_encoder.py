from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.EMPTY_TOK = "^"
        self.EMPTY_IND = 0
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        last_char = self.EMPTY_TOK

        for ind in inds:
            char = self.ind2char[ind]
            if ind != self.EMPTY_IND and char != last_char:
                result.append(char)
            last_char = char

        return ''.join(result)
    
    def _extend_and_merge(self, frame, state):
        new_state = defaultdict(float)

        for next_char_index, next_char_prob in enumerate(frame):
            for (prefix, last_char), prefix_prob in state.items():
                next_char = self.ind2char[next_char_index]

                if next_char != last_char and next_char != self.EMPTY_TOK:
                    # prefix is string -> immutable -> it's ok
                    prefix += next_char

                new_state[(prefix, next_char)] += prefix_prob * next_char_prob
                last_char = next_char

        return new_state
    
    def _truncate(self, state, beam_size):
        sorted_stats_list = sorted(list(state.items()), key=lambda x: -x[1])
        return dict(sorted_stats_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        # state[prefix, last_char] = prefix_prob
        state = {('', self.EMPTY_TOK): 1.0}

        for frame in probs:
            state = self._extend_and_merge(frame, state)
            state = self._truncate(state, beam_size)
        
        for (prefix, last_char), prefix_prob in state.items():
            hypos.append(Hypothesis(text=prefix, prob=prefix_prob))
        
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
