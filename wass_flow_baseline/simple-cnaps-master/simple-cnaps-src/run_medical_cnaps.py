"""
Medical evaluation wrapper for Simple CNAPS.

Evaluates a pretrained Simple CNAPS model on medical datasets from fsl_scoreboard/data/.
Uses Meta-Dataset's SingleDatasetReader for fixed-way/fixed-shot episodes.

Usage:
  cd fsl_scoreboard/simple-cnaps-master/simple-cnaps-src
  python run_medical_cnaps.py --subset 食管_1query --way 8 --shot 5 --query 1 \
      --n_runs 600 --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt \
      --data_path ../../records --output_json eval_result_medical.json
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
import torch

from simple_cnaps import SimpleCnaps
from meta_dataset_reader import SingleDatasetReader
from utils import print_and_log, get_log_files, aggregate_accuracy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MEDICAL_DATASETS = [
    '胃_1query',
    '十二指肠_1query',
    '食管_1query',
]

MEDICAL_WAYS = {
    '胃_1query': 8,
    '十二指肠_1query': 7,
    '食管_1query': 8,
}


class MedicalEvaluator:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, _, _ = get_log_files(
            os.path.join(self.args.checkpoint_dir, self.args.subset))
        print_and_log(self.logfile, "Options: %s\n" % self.args)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()

        tf.compat.v1.disable_eager_execution()
        self.session = tf.compat.v1.Session()

        self.metadataset = SingleDatasetReader(
            data_path=self.args.data_path,
            mode='test',
            dataset=self.args.subset,
            way=self.args.way,
            shot=self.args.shot,
            query_train=0,
            query_test=self.args.query,
        )

    def init_model(self):
        use_two_gpus = self.args.feature_adaptation == 'film+ar'
        model = SimpleCnaps(device=self.device, use_two_gpus=use_two_gpus, args=self.args).to(self.device)
        model.train()
        model.feature_extractor.eval()
        if use_two_gpus:
            model.distribute_model()
        return model

    def parse_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, required=True,
                            help='Path to Meta-Dataset records (e.g. ../../records)')
        parser.add_argument('--subset', type=str, required=True,
                            help='Medical subset name (e.g. 食管_1query)')
        parser.add_argument('--way', type=int, default=None,
                            help='Number of ways (default: auto from subset name)')
        parser.add_argument('--shot', type=int, default=5,
                            help='Number of support examples per class')
        parser.add_argument('--query', type=int, default=1,
                            help='Number of query examples per class')
        parser.add_argument('--n_runs', type=int, default=600,
                            help='Number of test episodes')
        parser.add_argument('--test_model_path', '-m', type=str, required=True,
                            help='Path to pretrained Simple CNAPS checkpoint')
        parser.add_argument('--feature_adaptation', type=str, default='film',
                            choices=['no_adaptation', 'film', 'film+ar'])
        parser.add_argument('--pretrained_resnet_path', type=str,
                            default='../model-checkpoints/pretrained_resnets/pretrained_resnet_meta_dataset.pt.tar',
                            help='Path to pretrained ResNet feature extractor')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/medical',
                            help='Directory for logs/checkpoints')
        parser.add_argument('--output_json', type=str, default='',
                            help='Path to save eval results as JSON')
        parser.add_argument('--shuffle_dataset', type=bool, default=True)
        args = parser.parse_args()

        if args.subset not in MEDICAL_DATASETS:
            parser.error(f'Unknown subset {args.subset}. Choose from: {MEDICAL_DATASETS}')
        if args.way is None:
            args.way = MEDICAL_WAYS[args.subset]

        return args

    def test(self):
        self.model.load_state_dict(torch.load(self.args.test_model_path))
        print_and_log(self.logfile, 'Testing model: %s\n' % self.args.test_model_path)

        accuracies = []
        for _ in range(self.args.n_runs):
            task_dict = self.metadataset.get_test_task(None)
            context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
            target_logits = self.model(context_images, context_labels, target_images)
            accuracy = aggregate_accuracy(target_logits, target_labels)
            accuracies.append(accuracy.item())
            del target_logits

        accuracies = np.array(accuracies)
        mean_acc = accuracies.mean() * 100.0
        confidence = (196.0 * accuracies.std()) / np.sqrt(len(accuracies))

        print_and_log(self.logfile, 'Dataset: %s\n' % self.args.subset)
        print_and_log(self.logfile, 'Result: %.2f +/- %.2f\n' % (mean_acc, confidence))
        print('Dataset: %s' % self.args.subset)
        print('Result: %.2f +/- %.2f (n_runs=%d)' % (mean_acc, confidence, self.args.n_runs))

        if self.args.output_json:
            result = {
                'dataset': self.args.subset,
                'method': 'SimpleCNAPS',
                'feature_adaptation': self.args.feature_adaptation,
                'n_ways': self.args.way,
                'n_shot': self.args.shot,
                'n_queries': self.args.query,
                'n_runs': self.args.n_runs,
                'test_acc': float(mean_acc),
                'pm': float(confidence),
                'test_model_path': self.args.test_model_path,
            }
            out_dir = os.path.dirname(self.args.output_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.args.output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print('Results saved to: %s' % self.args.output_json)

        self.logfile.close()
        return mean_acc, confidence

    def prepare_task(self, task_dict):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels


def main():
    evaluator = MedicalEvaluator()
    evaluator.test()


if __name__ == '__main__':
    main()
