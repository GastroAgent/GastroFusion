#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Huaxi OAT hyperparameter sensitivity directly from EXISTING fusion JSONs.

IMPORTANT
---------
This version does NOT require *_fsl_cache.json and does NOT rerun
fsl_mllm_fusion_tolast_rae_v2_huaxi.py.

It directly reads the three already-generated Huaxi latent+traj fusion files:

    INPUT_ROOT/
      十二指肠/Med-qwen3-vl-RL_latent_十二指肠_fusion_v2_traj.json
      食管/Med-qwen3-vl-RL_latent_食管_fusion_v2_traj.json
      胃/Med-qwen3-vl-RL_latent_胃_fusion_v2_traj.json

For every record it reuses the frozen evidence already stored in the fusion JSON:
    - ground_truth
    - mllm_score_map
    - fusion_v2_fsl_pred
    - fusion_v2_fsl_gap
    - fusion_v2_fsl_distance
    - fusion_v2_fsl_ranking

Then it reruns ONLY the fusion decision rule under different hyperparameters.

Protocol
--------
- Huaxi only
- latent + traj only
- RAE evidence is frozen from the existing fusion JSON
- One-at-a-time (OAT): only ONE parameter changes per run

Reference
---------
    stomach_alpha       = 0.25
    stomach_guard       = 0.01
    stomach_gap_thr     = 0.05

    esophagus_top1_thr  = 0.20
    esophagus_fsl_thr   = 0.10
    esophagus_gap_thr   = 0.01

    duodenum_top1_thr   = 0.80
    duodenum_fsl_thr    = 0.03

Outputs
-------
    sensitivity_huaxi_macro.csv
    sensitivity_huaxi_by_dataset.csv
    sensitivity_huaxi_runs.csv
    sensitivity_huaxi_reference.json

Each OAT run also writes:
    <sweep_root>/<param>__<value>/batch_fusion_v2_summary.csv
    <sweep_root>/<param>__<value>/batch_fusion_v2_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


# =============================================================================
# Paths
# =============================================================================

BASE = Path(
    "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃"
)

# This is now the EXISTING fusion-output root, not a cache/main-pair input root.
INPUT_ROOT = BASE / (
    "mllm_fsl_result/"
    "result_huaxi_831_grid/"
    "v2_fusion_traj_tlast_rae_latent_tsy"
)

DEFAULT_SWEEP_ROOT = (
    BASE
    / "mllm_fsl_result"
    / "sensitivity_v2_rae_latent_tsy_huaxi_91_new_reference"
)

ORGANS = ("胃", "食管", "十二指肠")


# =============================================================================
# Reference parameters / OAT grids
# =============================================================================

REFERENCE: Dict[str, float] = {
    "stomach_alpha": 0.25,
    "stomach_guard": 0.01,
    "stomach_gap_thr": 0.05,

    "esophagus_top1_thr": 0.20,
    "esophagus_fsl_thr": 0.10,
    "esophagus_gap_thr": 0.01,

    "duodenum_top1_thr": 0.80,
    "duodenum_fsl_thr": 0.03,
}

GRIDS: Dict[str, List[float]] = {
    # Stomach
    "stomach_alpha":       [0.05, 0.15, 0.25, 0.35, 0.45],
    "stomach_guard":       [0.005, 0.01, 0.02, 0.03, 0.05],
    "stomach_gap_thr":     [0.01, 0.03, 0.05, 0.07, 0.10],

    # Esophagus
    "esophagus_top1_thr":  [0.10, 0.15, 0.20, 0.25, 0.30],
    "esophagus_fsl_thr":   [0.03, 0.05, 0.10, 0.15, 0.20],
    "esophagus_gap_thr":   [0.005, 0.01, 0.02, 0.03, 0.05],

    # Duodenum
    "duodenum_top1_thr":   [0.70, 0.75, 0.80, 0.85, 0.90],
    "duodenum_fsl_thr":    [0.01, 0.02, 0.03, 0.05, 0.08],
}

PARAM_TO_ORGAN = {
    "stomach_alpha": "胃",
    "stomach_guard": "胃",
    "stomach_gap_thr": "胃",

    "esophagus_top1_thr": "食管",
    "esophagus_fsl_thr": "食管",
    "esophagus_gap_thr": "食管",

    "duodenum_top1_thr": "十二指肠",
    "duodenum_fsl_thr": "十二指肠",
}


def validate_sweep_configuration() -> None:
    """Ensure every OAT grid is aligned with the final reference setting."""
    reference_keys = set(REFERENCE)
    grid_keys = set(GRIDS)
    organ_keys = set(PARAM_TO_ORGAN)

    if reference_keys != grid_keys or reference_keys != organ_keys:
        raise ValueError(
            "REFERENCE, GRIDS and PARAM_TO_ORGAN must contain identical keys. "
            f"REFERENCE-only={sorted(reference_keys - grid_keys)}, "
            f"GRIDS-only={sorted(grid_keys - reference_keys)}, "
            f"missing-organ={sorted(reference_keys - organ_keys)}"
        )

    for param, reference_value in REFERENCE.items():
        if not any(
            abs(float(value) - reference_value) < 1e-12
            for value in GRIDS[param]
        ):
            raise ValueError(
                f"Sweep grid {param!r} does not include its reference value "
                f"{reference_value:g}: {GRIDS[param]}"
            )


# =============================================================================
# Utilities
# =============================================================================

def fmt_value(x: float) -> str:
    return f"{x:g}".replace("-", "m").replace(".", "p")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def normalize_class(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace(" ", "").strip().lower()


def class_eq(a: Any, b: Any) -> bool:
    return normalize_class(a) == normalize_class(b)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Exact V2 fusion decision logic
# =============================================================================

def _get_mllm_entry_exact(
    mllm_score_map: dict[str, Any],
    class_name: str,
) -> dict[str, Any]:
    entry = mllm_score_map.get(class_name)
    if isinstance(entry, dict):
        return entry

    target = normalize_class(class_name)
    for candidate, candidate_entry in mllm_score_map.items():
        if (
            normalize_class(candidate) == target
            and isinstance(candidate_entry, dict)
        ):
            return candidate_entry

    return {}


def softmax_from_distances(
    ranking: list[dict[str, Any]],
) -> dict[str, float]:
    ranking = [
        item
        for item in ranking
        if item.get("distance") is not None
    ]

    if not ranking:
        return {}

    negative = [
        -float(item["distance"])
        for item in ranking
    ]

    max_value = max(negative)
    exp_values = [
        math.exp(value - max_value)
        for value in negative
    ]

    total = sum(exp_values)
    if total <= 0:
        return {}

    return {
        str(ranking[i]["class"]):
        exp_values[i] / total
        for i in range(len(ranking))
    }


def predict_gap_guard_fusion(
    *,
    fsl_view: dict[str, Any],
    mllm_score_map: dict[str, Any],
    alpha: float,
    guard: float,
    gap_thr: float,
) -> tuple[str, dict[str, Any]]:
    """
    Exact V2 stomach rule:

      if gap >= gap_thr or MLLM_prob(FSL_top1) < guard:
          return FSL
      else:
          alpha * FSL_softmax + (1-alpha) * MLLM_prob
    """
    ranking = fsl_view["ranking"]
    fsl_top1 = fsl_view["top1"]
    gap = float(fsl_view["gap"])

    if not ranking or fsl_top1 == "未知":
        return fsl_top1, {
            "used_guard": True,
            "guard_reason": "invalid_fsl_view",
            "mllm_prob_on_fsl_top1": None,
            "combined_scores": {},
        }

    if not mllm_score_map:
        return fsl_top1, {
            "used_guard": True,
            "guard_reason": "empty_mllm_score_map",
            "mllm_prob_on_fsl_top1": None,
            "combined_scores": {},
        }

    mllm_fsl_entry = _get_mllm_entry_exact(
        mllm_score_map,
        fsl_top1,
    )
    mllm_fsl_prob = safe_float(
        mllm_fsl_entry.get("norm_prob"),
        1.0,
    )
    mllm_fsl_prob = float(
        1.0 if mllm_fsl_prob is None
        else mllm_fsl_prob
    )

    if gap >= gap_thr:
        return fsl_top1, {
            "used_guard": True,
            "guard_reason": "fsl_gap",
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
            "combined_scores": {},
        }

    if mllm_fsl_prob < guard:
        return fsl_top1, {
            "used_guard": True,
            "guard_reason": "mllm_prob_on_fsl_top1",
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
            "combined_scores": {},
        }

    fsl_scores = softmax_from_distances(ranking)

    best_class = fsl_top1
    best_score = -1.0
    combined_scores: dict[str, float] = {}

    for item in ranking:
        class_name = str(item["class"])

        fsl_score = fsl_scores.get(
            class_name,
            0.0,
        )

        entry = _get_mllm_entry_exact(
            mllm_score_map,
            class_name,
        )

        mllm_score = safe_float(
            entry.get("norm_prob"),
            0.0,
        )
        mllm_score = float(
            0.0 if mllm_score is None
            else mllm_score
        )

        score = (
            alpha * fsl_score
            + (1.0 - alpha) * mllm_score
        )

        combined_scores[class_name] = float(score)

        if score > best_score:
            best_score = score
            best_class = class_name

    return best_class, {
        "used_guard": False,
        "guard_reason": None,
        "mllm_prob_on_fsl_top1": mllm_fsl_prob,
        "combined_scores": combined_scores,
    }


def predict_high_conf_override(
    *,
    fsl_view: dict[str, Any],
    mllm_score_map: dict[str, Any],
    top1_thr: float,
    fsl_thr: float,
    gap_thr: float | None,
) -> tuple[str, dict[str, Any]]:
    """
    Exact V2 esophagus/duodenum rule.
    """
    fsl_top1 = fsl_view["top1"]
    gap = float(fsl_view["gap"])

    if not mllm_score_map:
        return fsl_top1, {
            "used_override": False,
            "override_reason": "empty_mllm_score_map",
            "mllm_top1_class": None,
            "mllm_top1_prob": None,
            "mllm_prob_on_fsl_top1": None,
        }

    valid_scores = {
        class_name: entry
        for class_name, entry in mllm_score_map.items()
        if (
            isinstance(entry, dict)
            and entry.get("norm_prob") is not None
            and safe_float(entry.get("norm_prob")) is not None
        )
    }

    if not valid_scores:
        return fsl_top1, {
            "used_override": False,
            "override_reason": "no_valid_mllm_scores",
            "mllm_top1_class": None,
            "mllm_top1_prob": None,
            "mllm_prob_on_fsl_top1": None,
        }

    top = max(
        valid_scores.items(),
        key=lambda x: float(x[1]["norm_prob"]),
    )

    mllm_top1_class = str(top[0])
    mllm_top1_prob = safe_float(
        top[1].get("norm_prob"),
        0.0,
    )
    mllm_top1_prob = float(
        0.0 if mllm_top1_prob is None
        else mllm_top1_prob
    )

    fsl_entry = _get_mllm_entry_exact(
        mllm_score_map,
        fsl_top1,
    )
    mllm_fsl_prob = safe_float(
        fsl_entry.get("norm_prob"),
        0.0,
    )
    mllm_fsl_prob = float(
        0.0 if mllm_fsl_prob is None
        else mllm_fsl_prob
    )

    gap_ok = (
        gap_thr is None
        or gap < gap_thr
    )

    use_override = (
        not class_eq(
            mllm_top1_class,
            fsl_top1,
        )
        and mllm_top1_prob > top1_thr
        and mllm_fsl_prob < fsl_thr
        and gap_ok
    )

    if use_override:
        return mllm_top1_class, {
            "used_override": True,
            "override_reason": "high_conf_mllm",
            "mllm_top1_class": mllm_top1_class,
            "mllm_top1_prob": mllm_top1_prob,
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
        }

    return fsl_top1, {
        "used_override": False,
        "override_reason": None,
        "mllm_top1_class": mllm_top1_class,
        "mllm_top1_prob": mllm_top1_prob,
        "mllm_prob_on_fsl_top1": mllm_fsl_prob,
    }


def params_for_organ(
    organ: str,
    params: Dict[str, float],
) -> dict[str, Any]:
    if organ == "胃":
        return {
            "strategy": "gap_guard_fusion",
            "alpha": float(params["stomach_alpha"]),
            "guard": float(params["stomach_guard"]),
            "gap_thr": float(params["stomach_gap_thr"]),
        }

    if organ == "食管":
        return {
            "strategy": "high_conf_override",
            "top1_thr": float(params["esophagus_top1_thr"]),
            "fsl_thr": float(params["esophagus_fsl_thr"]),
            "gap_thr": float(params["esophagus_gap_thr"]),
        }

    if organ == "十二指肠":
        return {
            "strategy": "very_high_conf_override",
            "top1_thr": float(params["duodenum_top1_thr"]),
            "fsl_thr": float(params["duodenum_fsl_thr"]),
            "gap_thr": None,
        }

    raise ValueError(f"Unsupported organ: {organ}")


def fusion_predict(
    *,
    fsl_view: dict[str, Any],
    mllm_score_map: dict[str, Any],
    organ: str,
    params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    strategy = params["strategy"]

    if strategy == "gap_guard_fusion":
        pred, diag = predict_gap_guard_fusion(
            fsl_view=fsl_view,
            mllm_score_map=mllm_score_map,
            alpha=float(params["alpha"]),
            guard=float(params["guard"]),
            gap_thr=float(params["gap_thr"]),
        )
        diag["strategy"] = strategy
        return pred, diag

    if strategy in (
        "high_conf_override",
        "very_high_conf_override",
    ):
        pred, diag = predict_high_conf_override(
            fsl_view=fsl_view,
            mllm_score_map=mllm_score_map,
            top1_thr=float(params["top1_thr"]),
            fsl_thr=float(params["fsl_thr"]),
            gap_thr=params.get("gap_thr"),
        )
        diag["strategy"] = strategy
        return pred, diag

    raise ValueError(
        f"Unknown strategy={strategy!r} for organ={organ}"
    )


# =============================================================================
# Existing fusion JSON discovery / validation
# =============================================================================

def discover_fusion_jsons(input_root: Path) -> Dict[str, Path]:
    """
    Require exactly one latent+traj fusion JSON for each Huaxi organ.
    Batch summary/failure JSONs are ignored automatically.
    """
    found: Dict[str, Path] = {}

    for organ in ORGANS:
        organ_root = input_root / organ
        if not organ_root.is_dir():
            raise FileNotFoundError(
                f"Missing organ directory: {organ_root}"
            )

        candidates = sorted(
            p for p in organ_root.rglob("*_fusion_v2_traj.json")
            if p.is_file()
            and "_latent_" in p.name.lower()
        )

        if len(candidates) != 1:
            raise RuntimeError(
                f"{organ}: expected exactly one latent *_fusion_v2_traj.json, "
                f"got {len(candidates)} under {organ_root}: "
                f"{[str(x) for x in candidates]}"
            )

        found[organ] = candidates[0]

    return found


def load_fusion_records(
    fusion_paths: Dict[str, Path],
) -> Dict[str, List[dict[str, Any]]]:
    all_records: Dict[str, List[dict[str, Any]]] = {}

    for organ, path in fusion_paths.items():
        data = load_json(path)
        if not isinstance(data, dict):
            raise TypeError(f"Fusion JSON top-level must be dict: {path}")

        records = data.get("results")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"Fusion JSON must contain non-empty results list: {path}"
            )

        required = (
            "ground_truth",
            "mllm_score_map",
            "fusion_v2_fsl_pred",
            "fusion_v2_fsl_gap",
            "fusion_v2_fsl_ranking",
        )

        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}: results[{idx}] is not a dict"
                )

            missing = [
                key for key in required
                if key not in record
            ]
            if missing:
                raise KeyError(
                    f"{path}: results[{idx}] missing fields required for "
                    f"direct sensitivity recomputation: {missing}"
                )

            ranking = record.get("fusion_v2_fsl_ranking")
            if not isinstance(ranking, list) or not ranking:
                raise ValueError(
                    f"{path}: results[{idx}].fusion_v2_fsl_ranking is empty/invalid"
                )

            score_map = record.get("mllm_score_map")
            if not isinstance(score_map, dict):
                raise TypeError(
                    f"{path}: results[{idx}].mllm_score_map is not dict"
                )

        all_records[organ] = records

    return all_records


# =============================================================================
# Recompute one organ and one OAT run
# =============================================================================

def recompute_organ(
    *,
    organ: str,
    records: List[dict[str, Any]],
    all_params: Dict[str, float],
    source_json: Path,
) -> tuple[dict[str, Any], List[dict[str, Any]]]:

    organ_params = params_for_organ(
        organ,
        all_params,
    )

    total = 0
    fsl_correct_count = 0
    mllm_correct_count = 0
    fusion_correct_count = 0
    fixed_count = 0
    broken_count = 0
    guard_count = 0
    override_count = 0

    detail_rows: List[dict[str, Any]] = []

    for idx, record in enumerate(records):
        ground_truth = record.get("ground_truth")

        ranking = record["fusion_v2_fsl_ranking"]
        fsl_top1 = str(record["fusion_v2_fsl_pred"])

        gap = safe_float(record.get("fusion_v2_fsl_gap"))
        if gap is None:
            if len(ranking) >= 2:
                d0 = safe_float(ranking[0].get("distance"))
                d1 = safe_float(ranking[1].get("distance"))
                if d0 is None or d1 is None:
                    raise ValueError(
                        f"{source_json}: record {idx} cannot recover FSL gap"
                    )
                gap = float(d1 - d0)
            else:
                gap = float("inf")

        distance = safe_float(
            record.get("fusion_v2_fsl_distance")
        )
        if distance is None:
            distance = safe_float(ranking[0].get("distance"))

        fsl_view = {
            "top1": fsl_top1,
            "gap": float(gap),
            "distance": distance,
            "ranking": ranking,
        }

        mllm_score_map = record["mllm_score_map"]

        fusion_pred, fusion_diag = fusion_predict(
            fsl_view=fsl_view,
            mllm_score_map=mllm_score_map,
            organ=organ,
            params=organ_params,
        )

        fsl_correct = class_eq(
            fsl_top1,
            ground_truth,
        )
        fusion_correct = class_eq(
            fusion_pred,
            ground_truth,
        )

        # MLLM itself is frozen and unaffected by the sweep.
        if "fusion_v2_mllm_correct" in record:
            mllm_correct = bool(
                record["fusion_v2_mllm_correct"]
            )
        elif "mllm_correct" in record:
            mllm_correct = bool(
                record["mllm_correct"]
            )
        else:
            valid_scores = {
                c: e for c, e in mllm_score_map.items()
                if isinstance(e, dict)
                and safe_float(e.get("norm_prob")) is not None
            }
            if valid_scores:
                mllm_top1 = max(
                    valid_scores.items(),
                    key=lambda x: float(x[1]["norm_prob"]),
                )[0]
                mllm_correct = class_eq(
                    mllm_top1,
                    ground_truth,
                )
            else:
                mllm_correct = False

        total += 1
        fsl_correct_count += int(fsl_correct)
        mllm_correct_count += int(mllm_correct)
        fusion_correct_count += int(fusion_correct)

        used_guard = bool(
            fusion_diag.get("used_guard", False)
        )
        used_override = bool(
            fusion_diag.get("used_override", False)
        )
        guard_count += int(used_guard)
        override_count += int(used_override)

        if (not fsl_correct) and fusion_correct:
            fixed_count += 1
        elif fsl_correct and (not fusion_correct):
            broken_count += 1

        detail_rows.append({
            "record_index": idx,
            "image_path": record.get("image_path")
                or record.get("query_path")
                or record.get("target_image"),
            "ground_truth": ground_truth,
            "fsl_pred": fsl_top1,
            "mllm_diag": record.get("mllm_diag"),
            "fusion_pred": fusion_pred,
            "fsl_correct": fsl_correct,
            "mllm_correct": mllm_correct,
            "fusion_correct": fusion_correct,
            "used_guard": used_guard,
            "used_override": used_override,
            "diagnostics": fusion_diag,
        })

    if total <= 0:
        raise RuntimeError(
            f"No records to recompute for {organ}: {source_json}"
        )

    summary = {
        "city": "华西",
        "organ": organ,
        "space": "latent",
        "fsl_mode": "traj",
        "strategy": organ_params["strategy"],
        "total": total,

        "fsl_correct": fsl_correct_count,
        "fsl_acc": fsl_correct_count / total,

        "mllm_correct": mllm_correct_count,
        "mllm_acc": mllm_correct_count / total,

        "fusion_correct": fusion_correct_count,
        "fusion_acc": fusion_correct_count / total,

        "fusion_minus_fsl_correct":
            fusion_correct_count - fsl_correct_count,
        "fusion_minus_mllm_correct":
            fusion_correct_count - mllm_correct_count,

        "fixed_fsl_errors": fixed_count,
        "broken_fsl_correct": broken_count,
        "net_fixed": fixed_count - broken_count,

        "guard_used": guard_count,
        "override_used": override_count,

        "alpha": organ_params.get("alpha"),
        "guard": organ_params.get("guard"),
        "top1_thr": organ_params.get("top1_thr"),
        "fsl_thr": organ_params.get("fsl_thr"),
        "gap_thr": organ_params.get("gap_thr"),

        "source_fusion_json": str(source_json),
        "recompute_policy":
            "reuse existing fusion_v2_fsl_ranking + mllm_score_map; "
            "rerun fusion decision only",
    }

    return summary, detail_rows


def run_one_configuration(
    *,
    params: Dict[str, float],
    records_by_organ: Dict[str, List[dict[str, Any]]],
    fusion_paths: Dict[str, Path],
    output_root: Path,
    save_recomputed_json: bool,
) -> List[dict[str, Any]]:

    summaries: List[dict[str, Any]] = []

    for organ in ORGANS:
        summary, details = recompute_organ(
            organ=organ,
            records=records_by_organ[organ],
            all_params=params,
            source_json=fusion_paths[organ],
        )
        summaries.append(summary)

        if save_recomputed_json:
            out_json = (
                output_root
                / organ
                / f"{organ}_recomputed_fusion_v2_traj.json"
            )
            atomic_write_json(
                out_json,
                {
                    "config": {
                        "city": "华西",
                        "organ": organ,
                        "space": "latent",
                        "fsl_mode": "traj",
                        "params": params_for_organ(
                            organ,
                            params,
                        ),
                        "source_fusion_json":
                            str(fusion_paths[organ]),
                    },
                    "summary": summary,
                    "results": details,
                },
            )

    summary_fields = [
        "city",
        "organ",
        "space",
        "fsl_mode",
        "strategy",
        "total",
        "fsl_correct",
        "fsl_acc",
        "mllm_correct",
        "mllm_acc",
        "fusion_correct",
        "fusion_acc",
        "fusion_minus_fsl_correct",
        "fusion_minus_mllm_correct",
        "fixed_fsl_errors",
        "broken_fsl_correct",
        "net_fixed",
        "guard_used",
        "override_used",
        "alpha",
        "guard",
        "top1_thr",
        "fsl_thr",
        "gap_thr",
        "source_fusion_json",
        "recompute_policy",
    ]

    write_csv(
        output_root / "batch_fusion_v2_summary.csv",
        summaries,
        summary_fields,
    )

    atomic_write_json(
        output_root / "batch_fusion_v2_summary.json",
        {
            "stage":
                "huaxi_v2_sensitivity_direct_from_existing_fusion_json",
            "input_root": str(INPUT_ROOT),
            "parameters": params,
            "dataset_count": len(summaries),
            "results": summaries,
        },
    )

    return summaries


# =============================================================================
# Summarize one OAT configuration
# =============================================================================

def summarize_one_run(
    *,
    sweep_param: str,
    sweep_value: float,
    params: Dict[str, float],
    summaries: List[dict[str, Any]],
    output_root: Path,
) -> tuple[dict, List[dict]]:

    target_rows = [
        r for r in summaries
        if str(r.get("city", "")).strip() == "华西"
        and str(r.get("space", "")).strip().lower() == "latent"
        and str(r.get("fsl_mode", "")).strip().lower() == "traj"
        and str(r.get("organ", "")).strip()
        in {"胃", "食管", "十二指肠"}
    ]

    if len(target_rows) != 3:
        raise RuntimeError(
            "Huaxi sensitivity expects exactly 3 latent/traj rows "
            f"(胃/食管/十二指肠), but got {len(target_rows)}"
        )

    expected_organ = PARAM_TO_ORGAN[sweep_param]
    organ_rows = [
        r for r in target_rows
        if str(r.get("organ", "")).strip() == expected_organ
    ]

    if len(organ_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one Huaxi row for organ={expected_organ}, "
            f"got {len(organ_rows)}"
        )

    fusion_all = [
        100.0 * float(r["fusion_acc"])
        for r in target_rows
    ]
    fsl_all = [
        100.0 * float(r["fsl_acc"])
        for r in target_rows
    ]
    mllm_all = [
        100.0 * float(r["mllm_acc"])
        for r in target_rows
    ]

    organ_row = organ_rows[0]

    macro_row = {
        "setting": "huaxi",
        "sweep_param": sweep_param,
        "sweep_organ": expected_organ,
        "sweep_value": sweep_value,
        "is_reference_value": int(
            abs(sweep_value - REFERENCE[sweep_param]) < 1e-12
        ),

        "macro_fusion_acc_percent": mean(fusion_all),
        "macro_fsl_acc_percent": mean(fsl_all),
        "macro_mllm_acc_percent": mean(mllm_all),
        "macro_fusion_minus_fsl_pp":
            mean(fusion_all) - mean(fsl_all),

        "organ_fusion_acc_percent":
            100.0 * float(organ_row["fusion_acc"]),
        "organ_fsl_acc_percent":
            100.0 * float(organ_row["fsl_acc"]),
        "organ_mllm_acc_percent":
            100.0 * float(organ_row["mllm_acc"]),
        "organ_fusion_minus_fsl_pp":
            100.0 * (
                float(organ_row["fusion_acc"])
                - float(organ_row["fsl_acc"])
            ),

        "fixed_fsl_errors_total":
            sum(int(r["fixed_fsl_errors"]) for r in target_rows),
        "broken_fsl_correct_total":
            sum(int(r["broken_fsl_correct"]) for r in target_rows),
        "net_fixed_total":
            sum(int(r["net_fixed"]) for r in target_rows),
        "guard_used_total":
            sum(int(r["guard_used"]) for r in target_rows),
        "override_used_total":
            sum(int(r["override_used"]) for r in target_rows),

        "organ_fixed_fsl_errors":
            int(organ_row["fixed_fsl_errors"]),
        "organ_broken_fsl_correct":
            int(organ_row["broken_fsl_correct"]),
        "organ_net_fixed":
            int(organ_row["net_fixed"]),
        "organ_guard_used":
            int(organ_row["guard_used"]),
        "organ_override_used":
            int(organ_row["override_used"]),

        "n_datasets": len(target_rows),
        "output_root": str(output_root),
    }

    for k, v in params.items():
        macro_row[k] = v

    dataset_rows: List[dict] = []
    for r in target_rows:
        dataset_rows.append({
            "setting": "huaxi",
            "sweep_param": sweep_param,
            "sweep_organ": expected_organ,
            "sweep_value": sweep_value,
            "is_reference_value":
                macro_row["is_reference_value"],

            "city": r["city"],
            "organ": r["organ"],
            "total": r["total"],

            "fsl_acc_percent":
                100.0 * float(r["fsl_acc"]),
            "mllm_acc_percent":
                100.0 * float(r["mllm_acc"]),
            "fusion_acc_percent":
                100.0 * float(r["fusion_acc"]),

            "fusion_minus_fsl_correct":
                int(r["fusion_minus_fsl_correct"]),
            "fixed_fsl_errors":
                int(r["fixed_fsl_errors"]),
            "broken_fsl_correct":
                int(r["broken_fsl_correct"]),
            "net_fixed":
                int(r["net_fixed"]),
            "guard_used":
                int(r["guard_used"]),
            "override_used":
                int(r["override_used"]),

            "source_fusion_json":
                r["source_fusion_json"],
            "output_root": str(output_root),
        })

    return macro_row, dataset_rows


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Huaxi OAT sensitivity directly from existing "
            "*_fusion_v2_traj.json files; no fsl_cache required."
        )
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        default=INPUT_ROOT,
        help=(
            "Existing Huaxi fusion result root containing "
            "胃/食管/十二指肠 subdirectories."
        ),
    )

    parser.add_argument(
        "--params",
        nargs="+",
        choices=sorted(GRIDS),
        default=list(GRIDS),
        help="OAT parameters to sweep. Default: all 8 parameters.",
    )

    parser.add_argument(
        "--sweep_root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
    )

    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help=(
            "Reuse one run when its batch_fusion_v2_summary.json exists."
        ),
    )

    parser.add_argument(
        "--save_recomputed_json",
        action="store_true",
        help=(
            "Also save per-query recomputed decisions for every OAT run. "
            "Default only writes run-level summaries to avoid large duplication."
        ),
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    validate_sweep_configuration()

    input_root = args.input_root.expanduser().resolve()
    sweep_root = args.sweep_root.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"INPUT_ROOT does not exist: {input_root}"
        )

    sweep_root.mkdir(parents=True, exist_ok=True)

    fusion_paths = discover_fusion_jsons(
        input_root
    )

    records_by_organ = load_fusion_records(
        fusion_paths
    )

    jobs = []
    for param in args.params:
        for value in GRIDS[param]:
            params = dict(REFERENCE)
            params[param] = float(value)

            output_root = (
                sweep_root
                / f"{param}__{fmt_value(float(value))}"
            )

            jobs.append(
                (
                    param,
                    float(value),
                    params,
                    output_root,
                )
            )

    print("=" * 110)
    print("[HUAXI / V2 FUSION / OAT SENSITIVITY / DIRECT FROM EXISTING FUSION JSON]")
    print(f"INPUT ROOT   = {input_root}")
    for organ in ORGANS:
        print(
            f"{organ:8s} = {fusion_paths[organ]} "
            f"({len(records_by_organ[organ])} records)"
        )
    print(f"SWEEP ROOT   = {sweep_root}")
    print(f"PARAMS       = {args.params}")
    print(f"JOBS         = {len(jobs)}")
    print("SPACE        = latent")
    print("FSL MODE     = traj")
    print("FSL EVIDENCE = frozen from fusion_v2_fsl_ranking")
    print("MLLM EVIDENCE= frozen from mllm_score_map")
    print("CACHE NEEDED = NO")
    print("=" * 110)

    macro_rows: List[dict] = []
    dataset_rows: List[dict] = []
    run_rows: List[dict] = []

    for idx, (
        param,
        value,
        params,
        output_root,
    ) in enumerate(jobs, 1):

        summary_json = (
            output_root / "batch_fusion_v2_summary.json"
        )

        print(
            f"\n[{idx:03d}/{len(jobs):03d}] "
            f"{param}={value:g} | "
            f"reference={REFERENCE[param]:g}"
        )

        status = "pending"
        error = ""
        summaries: List[dict[str, Any]] = []

        try:
            if args.dry_run:
                print(
                    f"[DRY] would recompute from {input_root} "
                    f"-> {output_root}"
                )
                status = "dry_run"

            elif args.reuse_existing and summary_json.is_file():
                reused = load_json(summary_json)
                summaries = reused.get("results", [])
                if len(summaries) != 3:
                    raise RuntimeError(
                        f"Invalid reused summary: expected 3 results, "
                        f"got {len(summaries)} in {summary_json}"
                    )
                print(f"[REUSE] {summary_json}")
                status = "reused"

            else:
                output_root.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                summaries = run_one_configuration(
                    params=params,
                    records_by_organ=records_by_organ,
                    fusion_paths=fusion_paths,
                    output_root=output_root,
                    save_recomputed_json=
                        bool(args.save_recomputed_json),
                )
                status = "ok"

        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {error}")

        run_rows.append({
            "setting": "huaxi",
            "sweep_param": param,
            "sweep_organ": PARAM_TO_ORGAN[param],
            "sweep_value": value,
            "reference_value": REFERENCE[param],
            "is_reference_value":
                int(abs(value - REFERENCE[param]) < 1e-12),
            "status": status,
            "error": error,
            "output_root": str(output_root),
        })

        if status in {"ok", "reused"}:
            macro_row, detail_rows = summarize_one_run(
                sweep_param=param,
                sweep_value=value,
                params=params,
                summaries=summaries,
                output_root=output_root,
            )

            macro_rows.append(macro_row)
            dataset_rows.extend(detail_rows)

            print(
                "[RESULT] "
                f"Huaxi Macro="
                f"{macro_row['macro_fusion_acc_percent']:.4f}% | "
                f"FSL Macro="
                f"{macro_row['macro_fsl_acc_percent']:.4f}% | "
                f"Gain="
                f"{macro_row['macro_fusion_minus_fsl_pp']:+.4f} pp | "
                f"{PARAM_TO_ORGAN[param]} Fusion="
                f"{macro_row['organ_fusion_acc_percent']:.4f}% | "
                f"{PARAM_TO_ORGAN[param]} FSL="
                f"{macro_row['organ_fsl_acc_percent']:.4f}% | "
                f"Fixed="
                f"{macro_row['organ_fixed_fsl_errors']} | "
                f"Broken="
                f"{macro_row['organ_broken_fsl_correct']}"
            )

    # =========================================================================
    # Write final sensitivity tables
    # =========================================================================

    run_fields = [
        "setting",
        "sweep_param",
        "sweep_organ",
        "sweep_value",
        "reference_value",
        "is_reference_value",
        "status",
        "error",
        "output_root",
    ]

    macro_fields = [
        "setting",
        "sweep_param",
        "sweep_organ",
        "sweep_value",
        "is_reference_value",

        "macro_fusion_acc_percent",
        "macro_fsl_acc_percent",
        "macro_mllm_acc_percent",
        "macro_fusion_minus_fsl_pp",

        "organ_fusion_acc_percent",
        "organ_fsl_acc_percent",
        "organ_mllm_acc_percent",
        "organ_fusion_minus_fsl_pp",

        "fixed_fsl_errors_total",
        "broken_fsl_correct_total",
        "net_fixed_total",
        "guard_used_total",
        "override_used_total",

        "organ_fixed_fsl_errors",
        "organ_broken_fsl_correct",
        "organ_net_fixed",
        "organ_guard_used",
        "organ_override_used",

        "n_datasets",

        *REFERENCE.keys(),

        "output_root",
    ]

    dataset_fields = [
        "setting",
        "sweep_param",
        "sweep_organ",
        "sweep_value",
        "is_reference_value",

        "city",
        "organ",
        "total",

        "fsl_acc_percent",
        "mllm_acc_percent",
        "fusion_acc_percent",

        "fusion_minus_fsl_correct",
        "fixed_fsl_errors",
        "broken_fsl_correct",
        "net_fixed",
        "guard_used",
        "override_used",

        "source_fusion_json",
        "output_root",
    ]

    runs_csv = (
        sweep_root
        / "sensitivity_huaxi_runs.csv"
    )

    macro_csv = (
        sweep_root
        / "sensitivity_huaxi_macro.csv"
    )

    dataset_csv = (
        sweep_root
        / "sensitivity_huaxi_by_dataset.csv"
    )

    reference_json_path = (
        sweep_root
        / "sensitivity_huaxi_reference.json"
    )

    write_csv(
        runs_csv,
        run_rows,
        run_fields,
    )

    write_csv(
        macro_csv,
        macro_rows,
        macro_fields,
    )

    write_csv(
        dataset_csv,
        dataset_rows,
        dataset_fields,
    )

    ref_rows = [
        r for r in macro_rows
        if r["is_reference_value"] == 1
    ]

    reference_json = {
        "protocol": "one-at-a-time",
        "setting": "huaxi",
        "space": "latent",
        "fsl_mode": "traj",
        "input_mode":
            "direct_from_existing_fusion_v2_traj_json",
        "input_root": str(input_root),
        "source_fusion_jsons": {
            organ: str(path)
            for organ, path in fusion_paths.items()
        },
        "cache_required": False,
        "reference_parameters": REFERENCE,
        "grids": GRIDS,
        "reference_rows": ref_rows,
        "note": (
            "FSL ranking/gap and MLLM score_map are frozen from the "
            "existing fusion JSONs. Only the fusion decision is rerun "
            "for OAT sensitivity."
        ),
    }

    atomic_write_json(
        reference_json_path,
        reference_json,
    )

    failed_rows = [
        r for r in run_rows
        if r["status"] == "failed"
    ]

    success_rows = [
        r for r in run_rows
        if r["status"] in {"ok", "reused"}
    ]

    dry_run_rows = [
        r for r in run_rows
        if r["status"] == "dry_run"
    ]

    print("\n" + "=" * 110)
    print("[HUAXI SWEEP SUMMARY]")
    print(f"total jobs = {len(run_rows)}")
    print(f"success    = {len(success_rows)}")
    print(f"failed     = {len(failed_rows)}")
    print(f"dry_run    = {len(dry_run_rows)}")
    print(f"runs       = {runs_csv}")
    print(f"macro      = {macro_csv}")
    print(f"dataset    = {dataset_csv}")
    print(f"reference  = {reference_json_path}")
    print("=" * 110)

    if args.dry_run:
        print(
            "\n[DRY RUN DONE] No fusion decisions were recomputed."
        )
        return

    if failed_rows:
        print("\n[FAILED RUNS]")
        for r in failed_rows:
            print(
                f"  {r['sweep_param']}="
                f"{r['sweep_value']} | "
                f"{r['error']}"
            )

        raise RuntimeError(
            f"{len(failed_rows)}/{len(run_rows)} "
            "Huaxi sensitivity jobs failed."
        )

    if not macro_rows:
        raise RuntimeError(
            "No successful Huaxi sensitivity results were summarized."
        )

    if len(success_rows) != len(macro_rows):
        raise RuntimeError(
            "Successful-run count does not match summarized macro-row count: "
            f"success={len(success_rows)}, macro_rows={len(macro_rows)}"
        )

    print("\n" + "=" * 110)
    print("[DONE]")
    print(f"macro     = {macro_csv}")
    print(f"dataset   = {dataset_csv}")
    print(f"runs      = {runs_csv}")
    print(f"reference = {reference_json_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
