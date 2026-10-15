"""Simple CNAPS medical evaluation with replacement-sampling support.

Supports test-test and test-train protocols.

test-test:
    support/context = test split
    query/target    = test split

test-train:
    support/context = train split
    query/target    = test split

The test-train implementation combines the context portion of a train task
with the target portion of a test task. To keep local labels semantically
aligned, test-train must cover every class in the selected medical records.
"""

import argparse
import json
import os
import random

os.environ.setdefault(
    "META_DATASET_ROOT",
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/"
    "fsl_scoreboard/simple-cnaps-master/meta-dataset",
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import torch

from simple_cnaps import SimpleCnaps
from meta_dataset_reader import SingleDatasetReader
from utils import print_and_log, get_log_files, aggregate_accuracy



MEDICAL_WAYS = {
    "胃_1query": 8,
    "十二指肠_1query": 7,
    "食管_1query": 8,
    "胃_7way_1query": 7,
    "十二指肠_7way_1query": 7,
    "食管_7way_1query": 7,
    "食管_8way_1query": 8,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


class MedicalEvaluator:
    def __init__(self):
        self.args = self.parse_command_line()
        self.set_seed(self.args.seed)

        log_dir = os.path.join(
            self.args.checkpoint_dir,
            self.args.eval_mode,
            self.args.subset,
        )
        self.checkpoint_dir, self.logfile, _, _ = get_log_files(log_dir)
        print_and_log(self.logfile, "Options: %s\n" % self.args)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.init_model()

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=session_config)

        # 'train_test' constructs both train and test pipelines. test-test
        # only needs the test pipeline.
        reader_mode = (
            "test" if self.args.eval_mode == "test-test" else "train_test"
        )
        self.metadataset = SingleDatasetReader(
            data_path=self.args.data_path,
            mode=reader_mode,
            dataset=self.args.subset,
            way=self.args.way,
            shot=self.args.shot,
            query_train=0,
            query_test=self.args.query,
            strict_replacement_sampling=self.args.sample_with_replacement,
            seed=self.args.seed,
        )

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        tf.compat.v1.set_random_seed(seed)

    def init_model(self):
        use_two_gpus = self.args.feature_adaptation == "film+ar"
        model = SimpleCnaps(
            device=self.device,
            use_two_gpus=use_two_gpus,
            args=self.args,
        ).to(self.device)
        model.train()
        model.feature_extractor.eval()
        if use_two_gpus:
            model.distribute_model()
        return model

    @staticmethod
    def parse_command_line():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", required=True)
        parser.add_argument("--subset", required=True)
        parser.add_argument(
            "--eval_mode",
            choices=["test-test", "test-train"],
            default="test-test",
        )
        parser.add_argument("--way", type=int, default=None)
        parser.add_argument("--shot", type=int, default=5)
        parser.add_argument("--query", type=int, default=1)
        parser.add_argument("--n_runs", type=int, default=100)
        parser.add_argument("--seed", type=int, default=12667)
        parser.add_argument(
            "--test_model_path", "-m", required=True
        )
        parser.add_argument(
            "--feature_adaptation",
            choices=["no_adaptation", "film", "film+ar"],
            default="film",
        )
        parser.add_argument(
            "--pretrained_resnet_path",
            default=(
                "../model-checkpoints/pretrained_resnets/"
                "pretrained_resnet_meta_dataset.pt.tar"
            ),
        )
        parser.add_argument(
            "--checkpoint_dir", default="./checkpoints/medical"
        )
        parser.add_argument("--output_json", default="")
        parser.add_argument(
            "--shuffle_dataset", type=str2bool, default=True
        )
        parser.add_argument(
            "--sample_with_replacement",
            type=str2bool,
            default=True,
            help=(
                "Allow repeated examples when a selected class contains "
                "fewer than shot + query images (default: true)."
            ),
        )
        args = parser.parse_args()

        if args.subset not in MEDICAL_WAYS:
            parser.error(
                "Unknown subset %s. Add its total class count to "
                "MEDICAL_WAYS first." % args.subset
            )
        total_classes = MEDICAL_WAYS[args.subset]
        if args.way is None:
            args.way = total_classes

        if args.eval_mode == "test-train" and args.way != total_classes:
            parser.error(
                "test-train with two stock SingleDatasetReader pipelines "
                "requires way == total subset classes so train/test local "
                "labels stay aligned. subset=%s has %d classes, way=%d."
                % (args.subset, total_classes, args.way)
            )
        return args

    def get_episode(self):
        if self.args.eval_mode == "test-test":
            return self.metadataset.get_test_task(None)

        # Support comes from train and query comes from test. With all classes
        # selected, both pipelines use the same dataset-local label mapping.
        train_task = self.metadataset.get_train_task()
        test_task = self.metadataset.get_test_task(None)

        task = {
            "context_images": train_task["context_images"],
            "context_labels": train_task["context_labels"],
            "target_images": test_task["target_images"],
            "target_labels": test_task["target_labels"],
        }
        self.validate_cross_split_task(task)
        return task

    def validate_cross_split_task(self, task):
        context_labels = np.asarray(task["context_labels"])
        target_labels = np.asarray(task["target_labels"])
        expected = set(range(self.args.way))
        context_classes = set(np.unique(context_labels).tolist())
        target_classes = set(np.unique(target_labels).tolist())

        if context_classes != expected or target_classes != expected:
            raise RuntimeError(
                "Cross-split class/label mismatch. "
                "context=%s, target=%s, expected=%s"
                % (
                    sorted(context_classes),
                    sorted(target_classes),
                    sorted(expected),
                )
            )
        expected_context = self.args.way * self.args.shot
        expected_target = self.args.way * self.args.query
        if len(context_labels) != expected_context:
            raise RuntimeError(
                "Expected %d support samples, got %d"
                % (expected_context, len(context_labels))
            )
        if len(target_labels) != expected_target:
            raise RuntimeError(
                "Expected %d query samples, got %d"
                % (expected_target, len(target_labels))
            )

    def test(self):
        state = torch.load(
            self.args.test_model_path, map_location=self.device
        )
        self.model.load_state_dict(state)
        print_and_log(
            self.logfile,
            "Testing model: %s\n" % self.args.test_model_path,
        )

        accuracies = []
        with torch.no_grad():
            for episode_id in range(self.args.n_runs):
                task_dict = self.get_episode()
                (
                    context_images,
                    target_images,
                    context_labels,
                    target_labels,
                ) = self.prepare_task(task_dict)
                target_logits = self.model(
                    context_images, context_labels, target_images
                )
                accuracy = aggregate_accuracy(
                    target_logits, target_labels
                )
                accuracies.append(accuracy.item())

                if (episode_id + 1) % 20 == 0:
                    print(
                        "Episode %d/%d"
                        % (episode_id + 1, self.args.n_runs)
                    )

        accuracies = np.asarray(accuracies, dtype=np.float64)
        mean_acc = accuracies.mean() * 100.0
        confidence = (
            1.96 * accuracies.std(ddof=0)
            / np.sqrt(len(accuracies))
            * 100.0
        )

        message = (
            "Dataset: %s | mode: %s | Result: %.2f +/- %.2f "
            "(n_runs=%d)\n"
            % (
                self.args.subset,
                self.args.eval_mode,
                mean_acc,
                confidence,
                self.args.n_runs,
            )
        )
        print_and_log(self.logfile, message)
        print(message.strip())

        output_json = self.args.output_json
        if not output_json:
            output_json = os.path.join(
                "./results_tsy",
                self.args.eval_mode,
                "%s_%dway_%dshot_%dquery_%dep.json"
                % (
                    self.args.subset,
                    self.args.way,
                    self.args.shot,
                    self.args.query,
                    self.args.n_runs,
                ),
            )
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        result = {
            "dataset": self.args.subset,
            "method": "SimpleCNAPS",
            "eval_mode": self.args.eval_mode,
            "support_split": (
                "test"
                if self.args.eval_mode == "test-test"
                else "train"
            ),
            "query_split": "test",
            "feature_adaptation": self.args.feature_adaptation,
            "n_ways": self.args.way,
            "n_shot": self.args.shot,
            "n_queries": self.args.query,
            "n_runs": self.args.n_runs,
            "seed": self.args.seed,
            "sample_with_replacement": self.args.sample_with_replacement,
            "support_query_disjoint_guaranteed": bool(
                self.args.sample_with_replacement
            ),
            "test_acc": float(mean_acc),
            "pm": float(confidence),
            "test_model_path": self.args.test_model_path,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Results saved to: %s" % output_json)

        self.session.close()
        self.logfile.close()
        return mean_acc, confidence

    def prepare_task(self, task_dict):
        context_images_np = np.asarray(task_dict["context_images"])
        context_labels_np = np.asarray(task_dict["context_labels"])
        target_images_np = np.asarray(task_dict["target_images"])
        target_labels_np = np.asarray(task_dict["target_labels"])

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        target_images_np = target_images_np.transpose([0, 3, 1, 2])

        if self.args.shuffle_dataset:
            context_perm = np.random.permutation(len(context_images_np))
            target_perm = np.random.permutation(len(target_images_np))
            context_images_np = context_images_np[context_perm]
            context_labels_np = context_labels_np[context_perm]
            target_images_np = target_images_np[target_perm]
            target_labels_np = target_labels_np[target_perm]

        context_images = torch.from_numpy(context_images_np).to(self.device)
        target_images = torch.from_numpy(target_images_np).to(self.device)
        context_labels = torch.from_numpy(context_labels_np).long().to(
            self.device
        )
        target_labels = torch.from_numpy(target_labels_np).long().to(
            self.device
        )
        return (
            context_images,
            target_images,
            context_labels,
            target_labels,
        )


def main():
    evaluator = MedicalEvaluator()
    evaluator.test()


if __name__ == "__main__":
    main()
