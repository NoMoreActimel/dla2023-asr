import numpy as np
from torch import nn

from hw_asr.base import BaseModel
from hw_asr.model.deepspeech2_conv2d import DeepSpeech2Conv2d
from hw_asr.model.deepspeech2_rnn_layer import DeepSpeech2RNNLayer


class DeepSpeech2Model(BaseModel):
    def __init__(
            self,
            freq_size,
            n_classes,
            conv2d_input_channels=1,
            conv2d_output_channels=32,
            conv2d_stride=(2, 2),
            conv2d_kernel_size=(41, 11),
            conv2d_relu_clipping_threshold=20,
            rnn_type='LSTM',
            n_rnn_layers=5,
            rnn_hidden_size=512,
            rnn_dropout_prob=0.1,
            **batch
    ):
        """
            Implementation of DeepSpeech2 model from https://arxiv.org/pdf/1512.02595.pdf
            This model classifies audio samples by log-spectrogram
            Expected input shape is Batch x Time x Freq
            :params:
            freq_size: number of frequencies - length by Freq axis
            n_class: number of classes
            rnn_type: type of RNN to use, must be either 'RNN', 'LSTM' or 'GRU'
            n_rnn_layers: number of RNN layers in final model
            rnn_hidden_size: size of hidden (freq) dimension in RNN layers
            rnn_dropout_prob: dropout probability in RNN layers
        """
        super().__init__()
        # input -> 
        # 2 InvariantConv2d layers by time and freq axis
        # 7 RNN layers
        # Lookahead Convolutions
        # Linear + Softmax ? 
        self.conv2d = DeepSpeech2Conv2d(
            input_channels=conv2d_input_channels,
            output_channels=conv2d_output_channels,
            stride=conv2d_stride,
            kernel_size=conv2d_kernel_size,
            relu_clipping_threshold=conv2d_relu_clipping_threshold
        )

        rnn_input_size = (freq_size * self.conv2d.kernel_size[0]) / self.conv2d.stride[0]
        rnn_input_size = int(np.floor(rnn_input_size)) + 1
        rnn_input_size = rnn_input_size * self.conv2d.output_channels

        self.rnns = nn.Sequential([
            DeepSpeech2RNNLayer(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                rnn_type=rnn_type,
                dropout_prob=rnn_dropout_prob
            ) for _ in range(n_rnn_layers)
        ])

        self.fc = nn.Linear(rnn_input_size, n_classes)


    def forward(self, spectrogram, input_lengths, **batch):
        input = torch.log(spectrogram)

        output, output_lengths = self.conv2d(input, input_lengths)
        output, output_lengths = self.rnns(output, output_lengths)
        output = self.fc(output).log_softmax(dim=-1)

        return {'logits': output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
