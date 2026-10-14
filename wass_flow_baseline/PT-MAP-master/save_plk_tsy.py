from __future__ import print_function

import argparse
import collections
import json
import os
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from data.datamgr import SimpleDataManager
import configs
import wrn_mixup_model
import res_mixup_model
from io_utils import get_resume_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract PT-MAP medical train/test features"
    )
    parser.add_argument(
        "--eval_mode",
        choices=["test-test", "test-train", "both"],
        default="both",
    )
    parser.add_argument(
        "--medical_subset",
        required=True,
        help="例如：十二指肠_7way_1query",
    )
    parser.add_argument(
        "--model",
        choices=["ResNet18", "WideResNet28_10"],
        default="ResNet18",
    )
    parser.add_argument("--method", default="S2M2_R")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="覆盖已经存在的特征文件",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="可选：显式指定训练权重目录",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="可选：显式指定保存 .plk 的目录",
    )
    return parser.parse_args()


class WrappedModel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def save_pickle(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(data), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_features(data_loader, model):
    model.eval()
    output_dict = collections.defaultdict(list)
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device, non_blocking=True)
            model_output = model(inputs)

            if isinstance(model_output, (tuple, list)):
                outputs = model_output[0]
            else:
                outputs = model_output

            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            for output, label in zip(outputs, labels):
                output_dict[int(label)].append(output)

            if (batch_id + 1) % 20 == 0:
                print(f"  processed batches: {batch_id + 1}")

    if not output_dict:
        raise RuntimeError("No features were extracted")

    return dict(output_dict)


def get_or_extract(
    output_path,
    data_loader,
    model,
    force_reextract,
):
    if os.path.isfile(output_path) and not force_reextract:
        print(f"[save_plk] exists, reuse:\n  {output_path}")
        return load_pickle(output_path)

    if os.path.isfile(output_path):
        print(f"[save_plk] overwrite:\n  {output_path}")

    features = extract_features(data_loader, model)
    save_pickle(output_path, features)
    print_feature_summary(output_path, features)
    return features


def print_feature_summary(path, features):
    class_counts = {
        int(class_id): len(values)
        for class_id, values in sorted(features.items())
    }
    total = sum(class_counts.values())
    print(f"[save_plk] saved:\n  {path}")
    print(f"  classes: {len(class_counts)}")
    print(f"  images : {total}")
    print(f"  counts : {class_counts}")


def read_num_classes(base_json):
    with open(base_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_names = data.get("label_names")
    if label_names is None:
        labels = data.get("image_labels", [])
        if not labels:
            raise ValueError(
                f"Cannot determine class count from {base_json}"
            )
        num_classes = len(set(labels))
        label_names = sorted(set(labels))
    else:
        num_classes = len(label_names)

    print(f"Num classes : {num_classes}")
    print(f"Classes     : {label_names}")
    return num_classes


def build_model(model_name, num_classes):
    if model_name == "WideResNet28_10":
        return wrn_mixup_model.wrn28_10(
            num_classes=num_classes
        )
    if model_name == "ResNet18":
        return res_mixup_model.resnet18(
            num_classes=num_classes
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no 'state' field: {checkpoint_path}"
        )

    state = checkpoint["state"]
    state_keys = list(state.keys())
    checkpoint_is_wrapped = bool(
        state_keys and state_keys[0].startswith("module.")
    )

    if checkpoint_is_wrapped:
        model = WrappedModel(model)

    current_state = model.state_dict()
    unexpected = [key for key in state if key not in current_state]
    mismatched = [
        key for key, value in state.items()
        if key in current_state
        and tuple(value.shape) != tuple(current_state[key].shape)
    ]
    if unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint/model mismatch:\n"
            f"  unexpected keys: {unexpected[:10]}\n"
            f"  shape mismatch : {mismatched[:10]}"
        )

    current_state.update(state)
    model.load_state_dict(current_state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    dataset = "medical"
    subset = args.medical_subset
    model_method = f"{args.model}_{args.method}"

    default_checkpoint_dir = os.path.join(
        configs.save_dir,
        "checkpoints",
        dataset,
        subset,
        model_method,
    )
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir
    output_dir = args.output_dir or os.path.join(
        default_checkpoint_dir,
        "last",
    )

    data_root = configs.data_dir["medical"]
    base_json = os.path.join(data_root, subset, "base.json")
    novel_json = os.path.join(data_root, subset, "novel.json")

    print("=" * 80)
    print("PT-MAP MEDICAL FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Evaluation mode : {args.eval_mode}")
    print(f"Subset          : {subset}")
    print(f"Model/method    : {model_method}")
    print(f"Device          : {device}")
    print(f"Checkpoint dir  : {checkpoint_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Base JSON       : {base_json}")
    print(f"Novel JSON      : {novel_json}")

    required_jsons = [novel_json]
    if args.eval_mode in ("test-train", "both"):
        required_jsons.append(base_json)
    for json_path in required_jsons:
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Dataset JSON not found:\n  {json_path}"
            )

    if not os.path.isfile(base_json):
        raise FileNotFoundError(
            "base.json is required to determine the checkpoint "
            f"classifier size:\n  {base_json}"
        )
    num_classes = read_num_classes(base_json)

    checkpoint_path = get_resume_file(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No trained checkpoint found in:\n  {checkpoint_dir}"
        )
    print(f"Checkpoint      : {checkpoint_path}")

    model = build_model(args.model, num_classes)
    model = load_checkpoint(model, checkpoint_path, device)
    print("[save_plk] checkpoint loaded successfully")

    data_manager = SimpleDataManager(
        args.image_size,
        batch_size=args.batch_size,
    )
    novel_loader = data_manager.get_data_loader(
        novel_json, aug=False
    )

    # novel/test features are used by both protocols.
    if args.eval_mode == "test-test":
        novel_features = get_or_extract(
            os.path.join(output_dir, "output.plk"),
            novel_loader,
            model,
            args.force_reextract,
        )
    else:
        novel_features = get_or_extract(
            os.path.join(output_dir, "novel_output.plk"),
            novel_loader,
            model,
            args.force_reextract,
        )

        if args.eval_mode == "both":
            # Keep output.plk for test-test without running the model twice.
            output_path = os.path.join(output_dir, "output.plk")
            if args.force_reextract or not os.path.isfile(output_path):
                save_pickle(output_path, novel_features)
                print_feature_summary(output_path, novel_features)
            else:
                print(f"[save_plk] exists, reuse:\n  {output_path}")

    if args.eval_mode in ("test-train", "both"):
        base_loader = data_manager.get_data_loader(
            base_json, aug=False
        )
        get_or_extract(
            os.path.join(output_dir, "base_output.plk"),
            base_loader,
            model,
            args.force_reextract,
        )

    print("=" * 80)
    print("Feature extraction finished")
    print("=" * 80)


if __name__ == "__main__":
    main()