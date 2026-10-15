#!/usr/bin/env python3
import argparse
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset

from backbone import ConvolutionalNeuralNetwork
from inner_loop_optimizer import gradient_update_parameters
from transform import get_transform


class EpisodeTrainFolder(Dataset):
    """Test-Train episodes.

    The GAP model/classifier always remains 8-way.  The dataset only assigns
    labels 0..effective_way-1 to classes that exist on both support and query
    sides.  Support images come from the source/train pool and query images
    come from the external test pool.
    """

    def __init__(self, support, test, ways, shots, query_shots, episodes, transform):
        self.support_root = support
        self.query_root = test
        self.ways = int(ways)                 # nominal organ-specific cap (8/8/7)
        self.shots = int(shots)
        self.query_shots = int(query_shots)
        self.episodes = int(episodes)
        self.transform = transform

        support_classes = sorted(
            x for x in os.listdir(support)
            if os.path.isdir(os.path.join(support, x))
        )
        self.support_data = {
            c: sorted(
                os.path.join(support, c, f)
                for f in os.listdir(os.path.join(support, c))
                if os.path.isfile(os.path.join(support, c, f))
            )
            for c in support_classes
        }

        self.query_data = {}
        if os.path.isdir(test):
            for c in sorted(os.listdir(test)):
                cdir = os.path.join(test, c)
                if not os.path.isdir(cdir):
                    continue
                self.query_data[c] = sorted(
                    os.path.join(cdir, f)
                    for f in os.listdir(cdir)
                    if os.path.isfile(os.path.join(cdir, f))
                )

        # Keep the existing reduced Test-Train semantics: use the support/query
        # class intersection and then cap by the requested organ-specific way.
        valid = [
            c for c in support_classes
            if c in self.query_data
            and len(self.support_data[c]) > 0
            and len(self.query_data[c]) > 0
        ]
        self.active_classes = valid[: self.ways]
        self.effective_way = len(self.active_classes)

        if self.effective_way == 0:
            raise RuntimeError("No matched support/query classes found")

    def __len__(self):
        return self.episodes

    def load(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    @staticmethod
    def _sample(paths, k):
        """Use distinct samples whenever possible; fall back to replacement."""
        if len(paths) >= k:
            return random.sample(paths, k)
        return random.choices(paths, k=k)

    def __getitem__(self, idx):
        random.seed(idx)
        sx, sy, qx, qy = [], [], [], []

        for y, c in enumerate(self.active_classes):
            support_paths = self._sample(self.support_data[c], self.shots)
            query_paths = self._sample(self.query_data[c], self.query_shots)

            sx.extend(self.load(p) for p in support_paths)
            sy.extend([y] * len(support_paths))
            qx.extend(self.load(p) for p in query_paths)
            qy.extend([y] * len(query_paths))

        return (
            torch.stack(sx),
            torch.tensor(sy, dtype=torch.long),
            torch.stack(qx),
            torch.tensor(qy, dtype=torch.long),
        )


def active_accuracy(logits, labels, effective_way):
    """Argmax only among active classes; the underlying classifier stays 8-way."""
    active_logits = logits[:, :effective_way]
    pred = torch.argmax(active_logits, dim=1)
    return (pred == labels).float().mean(), pred


def run(args):
    tf = get_transform(False)
    ds = EpisodeTrainFolder(
        args.support_dir,
        args.test_dir,
        args.episode_way,
        args.shots,
        args.query_shots,
        args.episodes,
        tf,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # IMPORTANT: model structure is fixed at 8-way.
    model = ConvolutionalNeuralNetwork(3, 8, hidden_size=args.hidden_size).to(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ck["model"], strict=True)
    M = ck["meta_params"]

    effective_way = ds.effective_way
    print("============================================================")
    print("[Protocol] Test-Train")
    print(f"[Protocol] model_output_way = 8 (unchanged)")
    print(f"[Protocol] nominal_episode_way = {args.episode_way}")
    print(f"[Protocol] effective_way = {effective_way}")
    print(f"[Protocol] active_classes = {ds.active_classes}")
    print(f"[Protocol] shots/query_shots = {args.shots}/{args.query_shots}")
    print("[Protocol] support CE and query argmax use only logits[:, :effective_way]")
    print("============================================================")

    acc = []
    invalid_full_top1 = 0
    total_queries = 0

    for i, (sx, sy, qx, qy) in enumerate(dl):
        sx, sy, qx, qy = [z.squeeze(0).to(args.device) for z in (sx, sy, qx, qy)]

        # Adaptation uses the same active candidate space as the episode, while
        # keeping the physical classifier at 8 outputs.
        support_logits = model(sx)
        support_active_logits = support_logits[:, :effective_way]
        loss = F.cross_entropy(support_active_logits, sy)

        params = gradient_update_parameters(
            model,
            M,
            loss,
            inner_lr=args.inner_lr,
            first_order=args.first_order,
            GAP=args.GAP,
            approx=args.approx,
        )

        with torch.no_grad():
            logits = functional_call(model, params, (qx,))  # still [Nq, 8]
            a, _ = active_accuracy(logits, qy, effective_way)

            # Diagnostic only: how often would unrestricted 8-way argmax choose
            # a class that does not exist in this evaluation episode?
            full_pred = torch.argmax(logits, dim=1)
            invalid_full_top1 += int((full_pred >= effective_way).sum().item())
            total_queries += int(qy.numel())

        acc.append(float(a.item()))
        if i < 3:
            print(
                f"[DEBUG] episode={i} support={tuple(sx.shape)} query={tuple(qx.shape)} "
                f"effective_way={effective_way} acc={a.item():.6f}"
            )

    acc_t = torch.tensor(acc, dtype=torch.float64)
    mean_acc = float(acc_t.mean().item())
    std_acc = float(acc_t.std(unbiased=True).item()) if len(acc) > 1 else 0.0
    ci = 1.96 * std_acc / (len(acc) ** 0.5)
    invalid_rate = invalid_full_top1 / total_queries if total_queries else 0.0

    print("============================================================")
    print(f"Test-Train Acc: {mean_acc:.6f}")
    print(f"CI95: {ci:.6f}")
    print(f"Episode Acc Std: {std_acc:.6f}")
    print(f"Unique Episode Accuracies: {len(set(acc))}")
    print(
        f"Diagnostic unrestricted-8way invalid-top1: "
        f"{invalid_full_top1}/{total_queries} ({100.0 * invalid_rate:.2f}%)"
    )
    print("============================================================")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True)
    p.add_argument("--support_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode_way", type=int, default=5)
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--query_shots", type=int, default=1)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--train_way", type=int, default=8)  # retained for CLI compatibility
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--inner_lr", type=float, default=0.01)
    p.add_argument("--first_order", action="store_true")
    p.add_argument("--GAP", action="store_true")
    p.add_argument("--approx", action="store_true")
    p.add_argument("--use_cuda", action="store_true")
    args = p.parse_args()
    args.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    run(args)
