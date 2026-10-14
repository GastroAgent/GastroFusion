##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
import argparse
import numpy as np
import torch
from datasets import miniImagenetOneShot
from option import Options
from experiments.OneShotMiniImageNetBuilder import miniImageNetBuilder
import tqdm
from logger import Logger


def main():
    args = Options().parse()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve data root
    # - miniimagenet: needs <dataroot>/{train,val,test}.csv and images/
    # - medical:      needs <dataroot>/<medical_subset>/{train,val,test}/<class>/*
    # When --dataset medical and --dataroot is left default (/tmp/omniglot),
    # we replace it with the project's data dir.
    if args.dataset == 'medical':
        if not args.dataroot or args.dataroot == '/tmp/omniglot':
            args.dataroot = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'data')
    print(f'[main] dataset={args.dataset}, dataroot={args.dataroot}, '
          f'medical_subset={args.medical_subset}, n_way={args.n_way}, '
          f'k_shot={args.k_shot}, k_query={args.k_query}, '
          f'fce={args.fce}, channels={args.channels}, '
          f'skip_test={args.skip_test}')

    log_dir = os.path.join(args.log_dir,
                           f'matching_{args.dataset}_{args.medical_subset}_'
                           f'n{args.n_way}k{args.k_shot}q{args.k_query}_'
                           f'fce{args.fce}_ch{args.channels}')
    logger = Logger(log_dir)

    # Build datasets (only train/val/test for miniimagenet & medical)
    dataTrain = miniImagenetOneShot.miniImagenetOneShotDataset(
        dataroot=args.dataroot, type='train',
        nEpisodes=args.total_train_batches * args.batch_size,
        classes_per_set=args.n_way, samples_per_class=args.k_shot,
        k_query=args.k_query,
        medical_subset=args.medical_subset, dataset=args.dataset,
        channels=args.channels)

    dataVal = miniImagenetOneShot.miniImagenetOneShotDataset(
        dataroot=args.dataroot, type='val',
        nEpisodes=args.total_val_batches * args.batch_size,
        classes_per_set=args.n_way, samples_per_class=args.k_shot,
        k_query=args.k_query,
        medical_subset=args.medical_subset, dataset=args.dataset,
        channels=args.channels)

    # When --skip_test is set, the user only wants training (no testing).
    # We still build a placeholder test dataset so that the DataLoader in
    # the builder has a non-None object, but we point it at the train split
    # (which always has enough classes) so it can construct episodes even
    # when the test split has fewer classes than n_way. We will never call
    # run_testing_epoch() during the training loop below.
    if args.skip_test:
        dataTest = miniImagenetOneShot.miniImagenetOneShotDataset(
            dataroot=args.dataroot, type='train',  # placeholder: use train
            nEpisodes=args.total_test_batches * args.batch_size,
            classes_per_set=args.n_way, samples_per_class=args.k_shot,
            k_query=args.k_query,
            medical_subset=args.medical_subset, dataset=args.dataset,
            channels=args.channels)
    else:
        dataTest = miniImagenetOneShot.miniImagenetOneShotDataset(
            dataroot=args.dataroot, type='test',
            nEpisodes=args.total_test_batches * args.batch_size,
            classes_per_set=args.n_way, samples_per_class=args.k_shot,
            k_query=args.k_query,
            medical_subset=args.medical_subset, dataset=args.dataset,
            channels=args.channels)

    builder = miniImageNetBuilder(
        dataTrain, dataVal, dataTest,
        save_dir=args.save_dir, dataset=args.dataset,
        medical_subset=args.medical_subset, save_best=args.save_best)
    builder.build_experiment(args.batch_size, args.n_way, args.k_shot,
                             args.channels, args.fce, lr=args.lr, wd=args.wd,
                             load_path=args.load)

    if args.eval_only:
        test_loss, test_acc = builder.run_testing_epoch()
        print(f'[Eval only] test_loss: {test_loss:.4f}, test_accuracy: {test_acc:.4f}')
        builder.save_result({
            'dataset': args.dataset,
            'medical_subset': args.medical_subset,
            'n_way': args.n_way, 'k_shot': args.k_shot, 'k_query': args.k_query,
            'fce': args.fce, 'channels': args.channels,
            'test_loss': test_loss, 'test_acc': test_acc,
        })
        return

    best_val = builder.current_best_val
    with tqdm.tqdm(total=args.total_epochs) as pbar_e:
        for e in range(args.total_epochs):
            train_loss, train_acc = builder.run_training_epoch()
            print(f'Epoch {e}: train_loss: {train_loss:.4f}, train_accuracy: {train_acc:.4f}')
            val_loss, val_acc = builder.run_validation_epoch()
            print(f'Epoch {e}: val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}')

            logger.log_value('train_loss', train_loss)
            logger.log_value('train_acc', train_acc)
            logger.log_value('val_loss', val_loss)
            logger.log_value('val_acc', val_acc)

            if val_acc >= best_val:
                best_val = val_acc
                if args.save_best:
                    builder.save_checkpoint(val_acc, tag='best')
                if args.skip_test:
                    # User opted out of testing; only log training/val.
                    test_loss, test_acc = float('nan'), float('nan')
                else:
                    test_loss, test_acc = builder.run_testing_epoch()
                print(f'Epoch {e}: test_loss: {test_loss:.4f}, test_accuracy: {test_acc:.4f}')
                logger.log_value('test_loss', test_loss)
                logger.log_value('test_acc', test_acc)
                # Also write the latest best result as json
                builder.save_result({
                    'best_epoch': e, 'best_val': val_acc,
                    'test_loss': test_loss, 'test_acc': test_acc,
                    'dataset': args.dataset,
                    'medical_subset': args.medical_subset,
                    'n_way': args.n_way, 'k_shot': args.k_shot,
                    'k_query': args.k_query,
                })
            pbar_e.update(1)
            logger.step()


if __name__ == '__main__':
    main()