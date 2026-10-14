##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Matching Network')

        # Path
        parser.add_argument('--dataroot', type=str, default='/tmp/omniglot',
                            help='path to dataset')
        parser.add_argument('--log-dir', default='./logs',
                            help='folder to output model checkpoints')
        parser.add_argument('--save-dir', default='./checkpoints',
                            help='folder to save best model')
        # Few-shot configuration
        parser.add_argument('--dataset', type=str, default='omniglot',
                            choices=['omniglot', 'miniimagenet', 'medical'],
                            help='which dataset')
        parser.add_argument('--n_way', type=int, default=5,
                            help='number of classes per episode (a.k.a. classes_per_set)')
        parser.add_argument('--k_shot', type=int, default=1,
                            help='number of support samples per class (a.k.a. samples_per_class)')
        parser.add_argument('--k_query', type=int, default=5,
                            help='number of query samples per class')
        # Medical data
        parser.add_argument('--medical_subset', type=str, default='胃_1query',
                            help='subdirectory under data/ that holds the medical images')
        # Training
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--fce', action='store_true', help='use full context embeddings (BiLSTM)')
        parser.add_argument('--channels', type=int, default=1,
                            help='1 for omniglot/medical-greyscale, 3 for medical-RGB')
        parser.add_argument('--total_epochs', type=int, default=500)
        parser.add_argument('--total_train_batches', type=int, default=1000)
        parser.add_argument('--total_val_batches', type=int, default=100)
        parser.add_argument('--total_test_batches', type=int, default=250)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--wd', type=float, default=1e-4)
        parser.add_argument('--seed', type=int, default=2191)
        # Inference / Test-only
        parser.add_argument('--eval_only', action='store_true',
                            help='skip training and run test only (requires --load)')
        parser.add_argument('--skip_test', action='store_true',
                            help='skip testing entirely during/after training '
                                 '(test split may not have enough classes); '
                                 'uses train split as a placeholder for DataLoader')
        parser.add_argument('--load', type=str, default='',
                            help='path to a saved checkpoint to load before training/testing')
        # Save
        parser.add_argument('--save_best', action='store_true',
                            help='save best val model to save-dir/dataset/<medical_subset or omniglot>/<run_name>.pt')

        parser.add_argument(
            '--eval_protocol',
            type=str,
            default='test-test',
            choices=['test-test', 'test-train'],
            help=(
                'Evaluation protocol: '
                'test-test means support=test/query=test; '
                'test-train means support=train/query=test'
            )
        )
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()