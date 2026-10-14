##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## Modified to support:
##   test-test:  support=test,  query=test
##   test-train: support=train, query=test
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
import argparse

import numpy as np
import torch
import tqdm

from datasets import miniImagenetOneShot_tsy as miniImagenetOneShot
from option import Options
from experiments.OneShotMiniImageNetBuilder import miniImageNetBuilder
from logger import Logger


def get_eval_splits(eval_protocol):
    """
    Protocol naming follows Query-Support order.

    test-test:
        query=test, support=test

    test-train:
        query=test, support=train
    """
    if eval_protocol == 'test-test':
        return 'test', 'test'

    if eval_protocol == 'test-train':
        return 'train', 'test'

    raise ValueError(
        f'Unsupported eval_protocol={eval_protocol}. '
        f'Expected "test-test" or "test-train".'
    )


def build_dataset(
    args,
    split_type,
    n_episodes,
    support_type=None,
    query_type=None,
):
    """Build one episodic dataset."""
    return miniImagenetOneShot.miniImagenetOneShotDataset(
        dataroot=args.dataroot,
        type=split_type,
        nEpisodes=n_episodes,
        classes_per_set=args.n_way,
        samples_per_class=args.k_shot,
        k_query=args.k_query,
        medical_subset=args.medical_subset,
        dataset=args.dataset,
        channels=args.channels,
        support_type=support_type,
        query_type=query_type,
        seed=args.seed,
    )


def main():
    args = Options().parse()

    # ============================================================
    # Reproducibility
    # ============================================================
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ============================================================
    # Resolve default medical data root
    # ============================================================
    if args.dataset == 'medical':
        if not args.dataroot or args.dataroot == '/tmp/omniglot':
            args.dataroot = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data'
            )

    # Avoid accidentally evaluating the placeholder train dataset.
    if args.eval_only and args.skip_test:
        raise ValueError(
            '--eval_only and --skip_test cannot both be enabled. '
            'For evaluation, use eval_only=True and skip_test=False.'
        )

    # ============================================================
    # Evaluation protocol
    # ============================================================
    support_type, query_type = get_eval_splits(
        args.eval_protocol
    )

    train_episodes = (
        args.total_train_batches * args.batch_size
    )
    val_episodes = (
        args.total_val_batches * args.batch_size
    )
    test_episodes = (
        args.total_test_batches * args.batch_size
    )

    print('=' * 70)
    print('[Main configuration]')
    print(f'dataset              : {args.dataset}')
    print(f'dataroot             : {args.dataroot}')
    print(f'medical_subset       : {args.medical_subset}')
    print(f'n_way                : {args.n_way}')
    print(f'k_shot               : {args.k_shot}')
    print(f'k_query              : {args.k_query}')
    print(f'batch_size           : {args.batch_size}')
    print(f'test episodes        : {test_episodes}')
    print(f'fce                  : {args.fce}')
    print(f'channels             : {args.channels}')
    print(f'eval_only            : {args.eval_only}')
    print(f'skip_test            : {args.skip_test}')
    print(f'eval_protocol        : {args.eval_protocol}')
    print(f'eval support split   : {support_type}')
    print(f'eval query split     : {query_type}')
    print(f'checkpoint           : {args.load}')
    print(f'save_dir             : {args.save_dir}')
    print(f'log_dir              : {args.log_dir}')
    print('=' * 70)

    # ============================================================
    # Output directories
    # ============================================================
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    protocol_save_dir = os.path.join(
        args.save_dir,
        args.eval_protocol,
        args.medical_subset
    )
    os.makedirs(protocol_save_dir, exist_ok=True)

    log_dir = os.path.join(
        args.log_dir,
        args.eval_protocol,
        (
            f'matching_{args.dataset}_{args.medical_subset}_'
            f'n{args.n_way}k{args.k_shot}q{args.k_query}_'
            f'fce{args.fce}_ch{args.channels}'
        )
    )
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(log_dir)

    # ============================================================
    # Build datasets
    # ============================================================
    if args.eval_only:
        # Pure evaluation only needs the requested test episodes.
        # Do not construct train-train/val-val here: their class/sample
        # counts may be insufficient for the evaluation n-way setting,
        # and they are never consumed by run_testing_epoch().
        dataTest = build_dataset(
            args=args,
            split_type='test',
            n_episodes=test_episodes,
            support_type=support_type,
            query_type=query_type,
        )

        # miniImageNetBuilder requires three dataset objects when it is
        # created. In eval-only mode its train and validation loaders are
        # unused, so reuse dataTest as a safe placeholder.
        dataTrain = dataTest
        dataVal = dataTest

        print(
            f'[Eval-only dataset] protocol={args.eval_protocol}, '
            f'support={support_type}, query={query_type}'
        )

    else:
        # Normal training still uses train-train episodes.
        dataTrain = build_dataset(
            args=args,
            split_type='train',
            n_episodes=train_episodes,
            support_type='train',
            query_type='train',
        )

        # Normal validation still uses val-val episodes.
        dataVal = build_dataset(
            args=args,
            split_type='val',
            n_episodes=val_episodes,
            support_type='val',
            query_type='val',
        )

        if args.skip_test:
            # Testing is disabled, so the builder can safely reuse the
            # already valid training dataset as its test placeholder.
            dataTest = dataTrain

            print(
                '[Test dataset] skip_test=True; '
                'reusing dataTrain as an unused placeholder.'
            )
        else:
            # test-test:  support=test,  query=test
            # test-train: support=train, query=test
            dataTest = build_dataset(
                args=args,
                split_type='test',
                n_episodes=test_episodes,
                support_type=support_type,
                query_type=query_type,
            )

            print(
                f'[Test dataset] protocol={args.eval_protocol}, '
                f'support={support_type}, query={query_type}'
            )

    # ============================================================
    # Build model and experiment
    # ============================================================
    builder = miniImageNetBuilder(
        dataTrain,
        dataVal,
        dataTest,
        save_dir=protocol_save_dir,
        dataset=args.dataset,
        medical_subset=args.medical_subset,
        save_best=args.save_best,
    )

    builder.build_experiment(
        args.batch_size,
        args.n_way,
        args.k_shot,
        args.channels,
        args.fce,
        lr=args.lr,
        wd=args.wd,
        load_path=args.load,
    )

    # ============================================================
    # Evaluation only
    # ============================================================
    if args.eval_only:
        if not args.load:
            raise ValueError(
                '--eval_only requires --load to specify '
                'a trained checkpoint.'
            )

        print('=' * 70)
        print('[Evaluation only]')
        print(f'Protocol     : {args.eval_protocol}')
        print(f'Support split: {support_type}')
        print(f'Query split  : {query_type}')
        print(f'Checkpoint   : {args.load}')
        print(f'Episodes     : {test_episodes}')
        print('=' * 70)

        test_loss, test_acc = builder.run_testing_epoch()

        print(
            f'[Evaluation completed] '
            f'test_loss={test_loss:.6f}, '
            f'test_accuracy={test_acc:.6f}, '
            f'test_accuracy_percent={test_acc * 100.0:.2f}%'
        )

        result = {
            'mode': 'eval_only',
            'dataset': args.dataset,
            'medical_subset': args.medical_subset,

            'eval_protocol': args.eval_protocol,
            'support_split': support_type,
            'query_split': query_type,

            'n_way': int(args.n_way),
            'k_shot': int(args.k_shot),
            'k_query': int(args.k_query),

            'batch_size': int(args.batch_size),
            'total_test_batches': int(
                args.total_test_batches
            ),
            'n_episodes': int(test_episodes),

            'seed': int(args.seed),
            'fce': bool(args.fce),
            'channels': int(args.channels),

            'dataroot': os.path.abspath(
                args.dataroot
            ),
            'checkpoint': os.path.abspath(
                args.load
            ),

            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'test_accuracy_percent': (
                float(test_acc) * 100.0
            ),
        }

        builder.save_result(result)

        # Also explicitly save a protocol-specific result file.
        result_json_path = os.path.join(
            protocol_save_dir,
            'result.json'
        )

        with open(
            result_json_path,
            'w',
            encoding='utf-8'
        ) as result_file:
            json.dump(
                result,
                result_file,
                ensure_ascii=False,
                indent=2
            )

        print(f'[Result saved] {result_json_path}')
        return

    # ============================================================
    # Training
    # ============================================================
    best_val = builder.current_best_val

    with tqdm.tqdm(
        total=args.total_epochs,
        desc='Training'
    ) as progress_bar:
        for epoch in range(args.total_epochs):
            # ----------------------------------------------------
            # Train
            # ----------------------------------------------------
            train_loss, train_acc = (
                builder.run_training_epoch()
            )

            print(
                f'Epoch {epoch}: '
                f'train_loss={train_loss:.4f}, '
                f'train_accuracy={train_acc:.4f}'
            )

            # ----------------------------------------------------
            # Validation
            # ----------------------------------------------------
            val_loss, val_acc = (
                builder.run_validation_epoch()
            )

            print(
                f'Epoch {epoch}: '
                f'val_loss={val_loss:.4f}, '
                f'val_accuracy={val_acc:.4f}'
            )

            logger.log_value(
                'train_loss',
                train_loss
            )
            logger.log_value(
                'train_acc',
                train_acc
            )
            logger.log_value(
                'val_loss',
                val_loss
            )
            logger.log_value(
                'val_acc',
                val_acc
            )

            # ----------------------------------------------------
            # Save/evaluate best validation checkpoint
            # ----------------------------------------------------
            if val_acc >= best_val:
                best_val = val_acc

                if args.save_best:
                    builder.save_checkpoint(
                        val_acc,
                        tag='best'
                    )

                if args.skip_test:
                    test_loss = float('nan')
                    test_acc = float('nan')
                else:
                    test_loss, test_acc = (
                        builder.run_testing_epoch()
                    )

                print(
                    f'Epoch {epoch}: '
                    f'test_loss={test_loss:.4f}, '
                    f'test_accuracy={test_acc:.4f}'
                )

                logger.log_value(
                    'test_loss',
                    test_loss
                )
                logger.log_value(
                    'test_acc',
                    test_acc
                )

                best_result = {
                    'mode': 'training',
                    'best_epoch': int(epoch),
                    'best_val': float(val_acc),

                    'dataset': args.dataset,
                    'medical_subset': (
                        args.medical_subset
                    ),

                    'eval_protocol': (
                        args.eval_protocol
                    ),
                    'support_split': (
                        support_type
                    ),
                    'query_split': (
                        query_type
                    ),

                    'n_way': int(args.n_way),
                    'k_shot': int(args.k_shot),
                    'k_query': int(args.k_query),

                    'batch_size': int(
                        args.batch_size
                    ),
                    'n_episodes': int(
                        test_episodes
                    ),
                    'seed': int(args.seed),

                    'checkpoint': (
                        args.load
                        if args.load
                        else None
                    ),

                    'test_loss': float(
                        test_loss
                    ),
                    'test_acc': float(
                        test_acc
                    ),
                    'test_accuracy_percent': (
                        float(test_acc) * 100.0
                    ),
                }

                builder.save_result(
                    best_result
                )

                best_result_json = os.path.join(
                    protocol_save_dir,
                    'result.json'
                )

                with open(
                    best_result_json,
                    'w',
                    encoding='utf-8'
                ) as result_file:
                    json.dump(
                        best_result,
                        result_file,
                        ensure_ascii=False,
                        indent=2
                    )

            progress_bar.update(1)
            logger.step()


if __name__ == '__main__':
    main()
