##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import unittest


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs):
        # inputs: [seq, batch, features]; matches inputs.device
        h0 = torch.zeros(self.lstm.num_layers * 2, inputs.size(1),
                         self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers * 2, inputs.size(1),
                         self.lstm.hidden_size)
        h0 = h0.to(inputs.device)
        c0 = c0.to(inputs.device)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn


class BidirectionalLSTMTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_accuracy(self):
        pass


if __name__ == '__main__':
    unittest.main()