from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech2RNNLayer(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=512,
            dropout_prob=0.1,
            rnn_type='LSTM'
    ):
        """
            Implementation of DeepSpeech2 1 Recurrent Neural Network layer
            Along with sequence-wise BatchNorm, as mentioned in the paper
            Supported types of RNNs: RNN, GRU, LSTM
            :params:
            input_size: input length by Freq axis, where input: Batch x Time x Freq
            type: 'RNN', 'LSTM' or 'GRU' - which RNN architecture to use
            hidden_size: size of hidden (freq) dimension in the RNN
            dropout_prob: probability of dropout after RNN layer
        """
        super().__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.activation = nn.ReLU()
        self.batchnorm1d = nn.BatchNorm1d(input_size)
        self.dropout_prob = dropout_prob

        print(f'rnn layer init started, input_size={input_size}')
        rnn_class = nn.RNN if self.rnn_type == 'RNN' else (nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU)
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1, # we apply sequence-wise batchnorm after each layer
            bidirectional=True,
            dropout=dropout_prob,
            batch_first=True,
            bias=True
        )
        
    def forward(self, input, input_lengths):
        # input: Batch x Time x Freq -> Batch x Freq x Time
        input = input.transpose(1, 2)

        # BatchNorm1d is applied on the Freq dim of prev-layer outputs
        output = self.activation(self.batchnorm1d(input))

        # output: Batch x Freq x Time -> Batch x Time x Freq
        output = output.transpose(1, 2)

        input = pack_padded_sequence(input, input_lengths, batch_first=True)
        output, output_lengths = self.rnn(input)
        output, output_lengths = pad_packed_sequence(output, total_length=input.shape[-1], batch_first=True)
        
        return output, output_lengths
