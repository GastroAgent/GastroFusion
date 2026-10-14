##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import unittest

import torch
import torch.nn as nn


class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Parameters
        ----------
        similarities:
            Cosine similarities with shape:

                [batch_size, support_size]

        support_set_y:
            One-hot labels of Support samples with shape:

                [batch_size, support_size, n_way]

        Returns
        -------
        preds:
            Class probabilities with shape:

                [batch_size, n_way]
        """

        if similarities.dim() == 1:
            similarities = similarities.unsqueeze(0)

        if similarities.dim() != 2:
            raise ValueError(
                'similarities must have shape [B, S], '
                f'but got {tuple(similarities.shape)}'
            )

        if support_set_y.dim() != 3:
            raise ValueError(
                'support_set_y must have shape [B, S, n_way], '
                f'but got {tuple(support_set_y.shape)}'
            )

        if similarities.size(0) != support_set_y.size(0):
            raise ValueError(
                'Batch-size mismatch: '
                f'similarities={tuple(similarities.shape)}, '
                f'support_set_y={tuple(support_set_y.shape)}'
            )

        if similarities.size(1) != support_set_y.size(1):
            raise ValueError(
                'Support-size mismatch: '
                f'similarities={tuple(similarities.shape)}, '
                f'support_set_y={tuple(support_set_y.shape)}'
            )

        # Attention must be normalized across Support samples,
        # i.e. along dimension S, not along batch dimension B.
        attention = torch.softmax(
            similarities,
            dim=1
        )

        # [B, 1, S] @ [B, S, n_way]
        # -> [B, 1, n_way]
        preds = torch.bmm(
            attention.unsqueeze(1),
            support_set_y.float()
        )

        # Remove only the singleton query dimension.
        # Do not use squeeze(), because that would also remove the
        # batch dimension when batch_size=1.
        preds = preds.squeeze(1)

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True

            print('\n[AttentionalClassify Debug]')
            print('similarities shape:', tuple(similarities.shape))
            print('similarities min:',
                similarities.detach().min().item())
            print('similarities max:',
                similarities.detach().max().item())
            print('similarities std:',
                similarities.detach().float().std().item())

            print('attention min:',
                attention.detach().min().item())
            print('attention max:',
                attention.detach().max().item())
            print('attention std:',
                attention.detach().float().std().item())
            print('attention row sum:',
                attention.detach().sum(dim=1).cpu())

            print('preds:',
                preds.detach().cpu())
            print('preds row sum:',
                preds.detach().sum(dim=1).cpu())

        return preds


class AttentionalClassifyTest(unittest.TestCase):
    def test_forward_batch_size_one(self):
        batch_size = 1
        support_size = 35
        n_way = 7

        similarities = torch.randn(
            batch_size,
            support_size
        )

        labels = torch.arange(n_way).repeat_interleave(5)

        support_set_y = torch.nn.functional.one_hot(
            labels,
            num_classes=n_way
        ).float().unsqueeze(0)

        classifier = AttentionalClassify()

        preds = classifier(
            similarities,
            support_set_y
        )

        self.assertEqual(
            tuple(preds.shape),
            (batch_size, n_way)
        )

        self.assertTrue(
            torch.allclose(
                preds.sum(dim=1),
                torch.ones(batch_size),
                atol=1e-6
            )
        )


if __name__ == '__main__':
    unittest.main()