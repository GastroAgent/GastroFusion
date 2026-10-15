import os
import argparse
import torch

from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Model-Agnostic Meta-Learning (MAML)'
    )

    parser.add_argument('--dataset', type=str, default='miniImageNet')

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--folder', type=str, default='./')

    parser.add_argument('--K_shots_for_support', type=int, default=5)
    parser.add_argument('--K_shots_for_query', type=int, default=1)

    parser.add_argument('--N_ways', type=int, default=8)
    parser.add_argument('--train_way', type=int, default=8)

    parser.add_argument('--train_num_update_step', type=int, default=5)
    parser.add_argument('--test_num_update_step', type=int, default=10)

    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=128)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--iter', type=int, default=60000)
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument('--outer_lr1', type=float, default=1e-4)
    parser.add_argument('--outer_lr2', type=float, default=1e-4)

    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--GAP', action='store_true')
    parser.add_argument('--approx', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.device = torch.device(
        'cuda' if args.use_cuda and torch.cuda.is_available()
        else 'cpu'
    )

    train(args)
