import numpy as np
import os
import glob
import argparse


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='miniImagenet',
                        help='CUB/miniImagenet/medical')
    parser.add_argument('--model', default='WideResNet28_10',
                        help='model: WideResNet28_10/ResNet18')
    parser.add_argument('--method', default='S2M2_R',
                        help='rotation/S2M2_R')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation during training')

    if script == 'train':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes')
        parser.add_argument('--save_freq', default=10, type=int,
                            help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int,
                            help='Starting epoch')
        parser.add_argument('--stop_epoch', default=400, type=int,
                            help='Stopping epoch')
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model')
        parser.add_argument('--lr', default=0.001, type=float,
                            help='learning rate')
        parser.add_argument('--batch_size', default=16, type=int,
                            help='batch size')
        parser.add_argument('--test_batch_size', default=2, type=int,
                            help='test batch size')
        parser.add_argument('--alpha', default=2.0, type=float,
                            help='beta distribution alpha for S2M2 training')
    elif script == 'test':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes')
        parser.add_argument('--n_shot', default=5, type=int,
                            help='number of support samples per class')
        parser.add_argument('--n_ways', default=5, type=int,
                            help='number of classes per episode')
        parser.add_argument('--n_queries', default=15, type=int,
                            help='number of query samples per class')
        parser.add_argument('--n_runs', default=10000, type=int,
                            help='number of episodes')
        parser.add_argument('--lam', default=10, type=float,
                            help='lambda for GaussianModel')
        parser.add_argument('--alpha', default=0.2, type=float,
                            help='alpha for MAP')
        parser.add_argument('--n_iter', default=20, type=int,
                            help='number of MAP iterations')
        parser.add_argument('--beta', default=0.5, type=float,
                            help='beta for power transform')

    # Used for medical / subset-aware loading.
    parser.add_argument('--medical_subset', type=str, default='',
                        help='medical subset name; appended to filelist path')
    parser.add_argument('--medical_root', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data',
                        help='root containing <medical_subset>/{train,val,test}')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)