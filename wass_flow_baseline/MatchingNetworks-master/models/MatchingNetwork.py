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
import torch.nn.functional as F

from models.BidirectionalLSTM import BidirectionalLSTM
from models.Classifier import Classifier
from models.DistanceNetwork import DistanceNetwork
from models.AttentionalClassify import AttentionalClassify


class MatchingNetwork(nn.Module):
    def __init__(
        self,
        keep_prob,
        batch_size=100,
        num_channels=1,
        learning_rate=0.001,
        fce=False,
        num_classes_per_set=5,
        num_samples_per_class=1,
        nClasses=0,
        image_size=28,
    ):
        super(MatchingNetwork, self).__init__()

        self.batch_size = batch_size
        self.fce = fce

        self.g = Classifier(
            layer_size=64,
            num_channels=num_channels,
            nClasses=nClasses,
            image_size=image_size,
        )

        if self.fce:
            self.lstm = BidirectionalLSTM(
                layer_sizes=[32],
                batch_size=self.batch_size,
                vector_dim=self.g.outSize,
            )

        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()

        self.keep_prob = keep_prob
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(
        self,
        support_set_images,
        support_set_labels_one_hot,
        target_image,
        target_label,
    ):
        """
        Parameters
        ----------
        support_set_images:
            Shape [B, S, C, H, W]

        support_set_labels_one_hot:
            Shape [B, S, n_way]

        target_image:
            Shape [B, Q, C, H, W]

        target_label:
            Shape [B, Q]

        Returns
        -------
        accuracy:
            Mean accuracy over all query positions.

        crossentropy_loss:
            Mean cross-entropy loss over all query positions.
        """

        # ========================================================
        # Validate dimensions
        # ========================================================
        if support_set_images.dim() != 5:
            raise ValueError(
                'support_set_images must have shape '
                f'[B, S, C, H, W], but got '
                f'{tuple(support_set_images.shape)}'
            )

        if target_image.dim() != 5:
            raise ValueError(
                'target_image must have shape '
                f'[B, Q, C, H, W], but got '
                f'{tuple(target_image.shape)}'
            )

        if support_set_labels_one_hot.dim() != 3:
            raise ValueError(
                'support_set_labels_one_hot must have shape '
                f'[B, S, n_way], but got '
                f'{tuple(support_set_labels_one_hot.shape)}'
            )

        # target_label may occasionally become one-dimensional when
        # batch_size=1 or Q=1. Restore it to [B, Q].
        if target_label.dim() == 0:
            target_label = target_label.view(1, 1)
        elif target_label.dim() == 1:
            target_label = target_label.unsqueeze(0)

        # ========================================================
        # Encode support images
        # ========================================================
        batch_size = support_set_images.size(0)
        support_size = support_set_images.size(1)

        support_flat = support_set_images.reshape(
            batch_size * support_size,
            *support_set_images.shape[2:]
        )

        g_support = self.g(support_flat)

        g_support = g_support.reshape(
            batch_size,
            support_size,
            -1
        )

        # ========================================================
        # Encode query images
        # ========================================================
        target_batch_size = target_image.size(0)
        query_size = target_image.size(1)

        if target_batch_size != batch_size:
            raise ValueError(
                'Support and query batch sizes do not match: '
                f'support batch={batch_size}, '
                f'query batch={target_batch_size}.'
            )

        target_flat = target_image.reshape(
            target_batch_size * query_size,
            *target_image.shape[2:]
        )

        g_target = self.g(target_flat)

        g_target = g_target.reshape(
            target_batch_size,
            query_size,
            -1
        )

        # ========================================================
        # Process every query position
        # ========================================================
        accuracy = g_target.new_tensor(0.0)
        crossentropy_loss = g_target.new_tensor(0.0)

        for query_index in range(query_size):
            # [B, feature_dim]
            query_embedding = g_target[
                :,
                query_index,
                :
            ]

            if self.fce:
                # g_support:
                #     [B, S, feature_dim]
                #
                # query_embedding.unsqueeze(1):
                #     [B, 1, feature_dim]
                #
                # Sequence after transpose:
                #     [S + 1, B, feature_dim]
                sequence = torch.cat(
                    [
                        g_support,
                        query_embedding.unsqueeze(1),
                    ],
                    dim=1,
                ).transpose(0, 1)

                sequence, _, _ = self.lstm(sequence)

                similarities = self.dn(
                    support_set=sequence[:-1],
                    input_image=sequence[-1],
                )

            else:
                similarities = self.dn(
                    support_set=g_support.transpose(0, 1),
                    input_image=query_embedding,
                )

            # Expected similarities shape: [S, B]
            similarities = similarities.transpose(0, 1)

            # Attentional classification.
            preds = self.classify(
                similarities,
                support_set_y=support_set_labels_one_hot,
            )

            # ====================================================
            # Fix batch_size=1 dimension collapse
            # ====================================================
            # AttentionalClassify may return:
            #
            #     batch_size > 1: [B, n_way]
            #     batch_size = 1: [n_way]
            #
            # CrossEntropyLoss and max(dim=1) both require:
            #
            #     [B, n_way]
            if preds.dim() == 1:
                preds = preds.unsqueeze(0)

            if preds.dim() != 2:
                raise RuntimeError(
                    'Expected predictions with shape [B, n_way], '
                    f'but got {tuple(preds.shape)}.'
                )

            if preds.size(0) != batch_size:
                raise RuntimeError(
                    'Prediction batch size does not match the input: '
                    f'preds batch={preds.size(0)}, '
                    f'input batch={batch_size}.'
                )

            # Always keep target as [B].
            query_target = target_label[
                :,
                query_index
            ].long().reshape(-1)

            if query_target.numel() != batch_size:
                raise RuntimeError(
                    'Target batch size does not match the input: '
                    f'target size={query_target.numel()}, '
                    f'input batch={batch_size}.'
                )

            # Class prediction along n_way dimension.
            _, predicted_indices = preds.max(dim=1)

            query_accuracy = (
                predicted_indices.reshape(-1)
                == query_target
            ).float().mean()

            query_loss = F.cross_entropy(
                preds,
                query_target,
            )

            accuracy = accuracy + query_accuracy
            crossentropy_loss = (
                crossentropy_loss + query_loss
            )

        return (
            accuracy / query_size,
            crossentropy_loss / query_size,
        )


class MatchingNetworkTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_accuracy(self):
        pass


if __name__ == '__main__':
    unittest.main()