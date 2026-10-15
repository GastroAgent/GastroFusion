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


class EpisodeFolder(Dataset):
    """Test-Test episodes with disjoint support/query images.

    The GAP classifier always remains 8-way.  For a given centre-organ cohort,
    only classes that can provide `shots + query_shots` distinct images are
    active.  Labels are mapped to 0..effective_way-1 in sorted class order.
    """

    def __init__(self, root, ways, shots, query_shots, episodes, transform):
        self.root = root
        self.ways = int(ways)                 # nominal organ-specific cap (8/8/7)
        self.shots = int(shots)
        self.query_shots = int(query_shots)
        self.episodes = int(episodes)
        self.transform = transform

        self.classes = sorted(
            x for x in os.listdir(root)
            if os.path.isdir(os.path.join(root, x))
        )
        self.data = {
            c: sorted(
                os.path.join(root, c, f)
                for f in os.listdir(os.path.join(root, c))
                if os.path.isfile(os.path.join(root, c, f))
            )
            for c in self.classes
        }

        # Strict Test-Test: query must not appear in support.  Therefore each
        # active class must have enough distinct images for K-shot + query.
        need = self.shots + self.query_shots
        valid = [c for c in self.classes if len(self.data[c]) >= need]
        self.active_classes = valid[: self.ways]
        self.effective_way = len(self.active_classes)

        if self.effective_way == 0:
            raise RuntimeError(
                f"No valid Test-Test classes with at least {need} distinct images under {root}"
            )

    def __len__(self):
        return self.episodes

    def load(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        random.seed(idx)
        sx, sy, qx, qy = [], [], [], []

        for y, c in enumerate(self.active_classes):
            # One joint draw guarantees support/query disjointness.
            chosen = random.sample(self.data[c], self.shots + self.query_shots)
            support_paths = chosen[: self.shots]
            query_paths = chosen[self.shots :]

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
    ds = EpisodeFolder(
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
    print("[Protocol] Test-Test")
    print(f"[Protocol] model_output_way = 8 (unchanged)")
    print(f"[Protocol] nominal_episode_way = {args.episode_way}")
    print(f"[Protocol] effective_way = {effective_way}")
    print(f"[Protocol] active_classes = {ds.active_classes}")
    print(f"[Protocol] shots/query_shots = {args.shots}/{args.query_shots}")
    print("[Protocol] support/query images are strictly disjoint")
    print("[Protocol] support CE and query argmax use only logits[:, :effective_way]")
    print("============================================================")

    acc = []
    invalid_full_top1 = 0
    total_queries = 0

    for i, (sx, sy, qx, qy) in enumerate(dl):
        sx, sy, qx, qy = [z.squeeze(0).to(args.device) for z in (sx, sy, qx, qy)]

        # Adaptation uses only the active candidate outputs; classifier remains 8-way.
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
    print(f"Test-Test Acc: {mean_acc:.6f}")
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
