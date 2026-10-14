#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate an F1-v10c checkpoint under the simplified test-train protocol.

Protocol
--------
* Query comes from the TEST pool.
* Support comes from a separate TRAIN/support pool.
* Candidate classes are the query/support class intersection.
* A class needs >=1 test/query image and >=1 train/support image.
* K-shot support is sampled without replacement when enough train images exist,
  otherwise with replacement.  The test query is NOT explicitly excluded from the
  train support pool (this is the intentionally simplified test-train setting).
* One N-way / K-shot / 1-query episode is sampled per episode.
* By default all shared classes are preserved as the candidate set (dynamic-way).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

# Reuse the checkpoint-faithful evaluator/runtime from the strict test-test script.
from test_f1_v10c_testtest import (
    PRETRAIN_BASENAME,
    EpisodeSpec,
    _cfg,
    _load_pretrain_module,
    _torch_load,
    effective_way,
    evaluate_specs,
    load_runtime,
    normalize_pool,
    sample_k,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("F1-v10c simplified test-train evaluation")
    p.add_argument("--pretrain_py", type=str, default=None,
                   help=f"Path to {PRETRAIN_BASENAME}; default: sibling file next to test script")
    p.add_argument("--f1_checkpoint", type=str, required=True)
    p.add_argument("--test_class_json_root", type=str, required=True, help="TEST/query pool")
    p.add_argument("--train_class_json_root", type=str, required=True, help="TRAIN/support pool")
    p.add_argument("--output_json", type=str, required=True)

    p.add_argument("--project_root", type=str, default=None)
    p.add_argument("--m2_checkpoint", type=str, default=None)
    p.add_argument("--dit_checkpoint", type=str, default=None)

    p.add_argument("--n_way", type=int, default=0,
                   help="<=0 preserves all shared query/support classes; positive value caps the way")
    p.add_argument("--n_shot", type=int, default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    p.add_argument("--pair_chunk_size", type=int, default=None)
    p.add_argument("--rae_chunk_size", type=int, default=None)
    p.add_argument("--correction_topk", type=int, default=None)
    p.add_argument("--metric_device", type=str, default="cuda:0")
    p.add_argument("--flow_device", type=str, default="cuda:0")
    p.add_argument("--rae_device", type=str, default="cuda:0")
    p.add_argument("--save_episode_details", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()


def build_testtrain_episode(
    query_pool: dict[str, list[str]], support_pool: dict[str, list[str]], *,
    eligible_classes: list[str], n_way: int, n_shot: int, rng: random.Random,
) -> EpisodeSpec:
    if n_way == len(eligible_classes):
        classes = list(eligible_classes)
        rng.shuffle(classes)
    else:
        classes = rng.sample(eligible_classes, n_way)
        rng.shuffle(classes)

    query_class = rng.choice(classes)
    query_path = rng.choice(query_pool[query_class])
    support_rows, replacement_flags = [], []
    for c in classes:
        # Intentionally NO exact-query exclusion in simplified test-train.
        row, used_replacement = sample_k(support_pool[c], n_shot, rng)
        support_rows.append(row)
        replacement_flags.append(used_replacement)
    return EpisodeSpec(
        classes=classes,
        support_paths=support_rows,
        query_path=query_path,
        query_class=query_class,
        target_index=classes.index(query_class),
        support_used_replacement=replacement_flags,
    )


def run_testtrain(cli: argparse.Namespace) -> dict[str, Any]:
    mod, pretrain_path = _load_pretrain_module(cli.pretrain_py)
    ckpt = _torch_load(cli.f1_checkpoint)
    if not isinstance(ckpt, dict):
        raise ValueError("F1 checkpoint must be a dict")
    if hasattr(mod, "F1V10_STAGE") and ckpt.get("stage") != mod.F1V10_STAGE:
        raise ValueError(f"checkpoint stage={ckpt.get('stage')!r} is not F1-v10c stage={mod.F1V10_STAGE!r}")
    if ckpt.get("class_feature_names") not in (None, list(mod.V10_CLASS_FEATURE_NAMES)):
        raise ValueError("checkpoint class_feature_names do not match V10 evaluator")

    query_pool = normalize_pool(mod.load_class_pool(None, cli.test_class_json_root))
    support_pool = normalize_pool(mod.load_class_pool(None, cli.train_class_json_root))
    eligible = sorted([
        c for c in set(query_pool).intersection(support_pool)
        if len(query_pool[c]) >= 1 and len(support_pool[c]) >= 1
    ])
    query_only = sorted(set(query_pool) - set(support_pool))
    support_only = sorted(set(support_pool) - set(query_pool))
    n_way = effective_way(cli.n_way, len(eligible))

    saved = dict(ckpt.get("args") or {})
    n_shot = int(cli.n_shot if cli.n_shot is not None else _cfg(saved, "n_shot", 5))
    if n_shot < 1:
        raise ValueError("n_shot must be >=1")
    if cli.episodes < 1:
        raise ValueError("episodes must be >=1")

    print("=" * 120)
    print("[F1-v10c TEST-TRAIN] simplified separate-support-pool evaluation")
    print(f"query/test pool={cli.test_class_json_root}")
    print(f"support/train pool={cli.train_class_json_root}")
    print(f"shared eligible classes={len(eligible)}, effective n_way={n_way}")
    print(f"episode={n_way}-way {n_shot}-shot 1-query x {cli.episodes}")
    print("support rule=train pool; no strict query exclusion; without replacement if enough else with replacement")
    print("=" * 120)

    rng = random.Random(cli.seed)
    specs = [
        build_testtrain_episode(
            query_pool, support_pool, eligible_classes=eligible,
            n_way=n_way, n_shot=n_shot, rng=rng,
        )
        for _ in range(cli.episodes)
    ]

    runtime = load_runtime(cli, mod, ckpt, n_way=n_way, n_shot=n_shot)
    args, metric, flow, rae, corrector, shot_tau, cls_tau, transform, metric_device, flow_device, rae_device, paths = runtime
    metrics = evaluate_specs(
        specs, mod=mod, args=args, metric=metric, flow=flow, rae=rae, corrector=corrector,
        shot_tau=shot_tau, cls_tau=cls_tau, transform=transform,
        metric_device=metric_device, rae_device=rae_device,
        episode_batch_size=cli.episode_batch_size, seed=cli.seed,
        save_episode_details=cli.save_episode_details,
    )

    result = {
        "stage": "f1-v10c-eval-test-train-simplified",
        "protocol": "test-train",
        "query_source": str(cli.test_class_json_root),
        "support_source": str(cli.train_class_json_root),
        "candidate_rule": "full test-query/train-support class intersection; optional n_way cap; support-only distractors excluded",
        "support_sampling": "train/support pool; without_replacement_if_enough_else_with_replacement; no strict query exclusion",
        "query_support_exact_overlap_allowed": True,
        "eligible_classes": eligible,
        "query_only_classes_excluded": query_only,
        "support_only_classes_excluded": support_only,
        "n_way_requested": int(cli.n_way),
        "n_way_eval": int(n_way),
        "n_shot": int(n_shot),
        "n_query": 1,
        "episodes_requested": int(cli.episodes),
        "seed": int(cli.seed),
        "num_steps": int(args.num_steps),
        "solver": args.solver,
        "correction_topk": int(args.correction_topk),
        "shot_tau": float(shot_tau.item()),
        "cls_tau": float(cls_tau.item()),
        "f1_checkpoint": str(cli.f1_checkpoint),
        "pretrain_py": pretrain_path,
        **paths,
        "metrics": metrics,
    }
    out = Path(cli.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "episode_details"}, ensure_ascii=False, indent=2))
    print(f"[saved] {out}")
    return result


def main() -> None:
    cli = parse_args()
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    run_testtrain(cli)


if __name__ == "__main__":
    main()
