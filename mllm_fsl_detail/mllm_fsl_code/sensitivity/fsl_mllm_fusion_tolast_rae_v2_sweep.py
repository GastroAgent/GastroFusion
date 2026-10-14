#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensitivity analysis for fsl_mllm_fusion_tolast_rae_v2.py

Protocol:
- One-at-a-time (OAT) sensitivity analysis.
- Only one hyperparameter changes at a time.
- All other parameters stay at the final/default setting:
    Stomach:
        alpha=0.25, guard=0.01, gap_thr=0.05
    Esophagus:
        top1_thr=0.20, fsl_thr=0.10, gap_thr=0.01
    Duodenum:
        top1_thr=0.80, fsl_thr=0.03

- The underlying V2 fusion Python is NOT modified.
- Each run receives a unique output_root.
- Summary is computed from batch_fusion_summary.csv.
- By default, only `space == latent` rows are used for paper sensitivity results.
- Supports both external settings:
    few_way
    lot_way

Outputs:
    <SWEEP_ROOT>/
        sensitivity_runs.csv
        sensitivity_macro.csv
        sensitivity_by_dataset.csv
        sensitivity_best_reference.json
        few_way/...
        lot_way/...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# =============================================================================
# 1. Fixed paths
# =============================================================================

V2_PY = Path(
    "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/"
    "mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2.py"
)

BASE = Path(
    "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃"
)

RAE_BASE = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831/external"
)

SWEEP_ROOT = (
    BASE
    / "mllm_fsl_result"
    / "sensitivity_v2_rae_latent_tsy_external_91_new_reference"
)


SETTINGS = {
    "few_way": {
        "input_root": BASE / "mllm_fsl_result/mllm_fsl_few_way_subset_91_grid/v2_fusion_traj_tlast_rae_latent",
        "raw_fsl_result_root": BASE / "eval_tsy_image_space/mllm_fsl_subset_align_old_test_few_way/result",
        "rae_fsl_result_root": RAE_BASE / "mllm_fsl_subset_agent_few_way_91",
    },
    "lot_way": {
        "input_root": BASE / "mllm_fsl_result/mllm_fsl_lot_way_subset_91_grid/v2_fusion_traj_tlast_rae_latent",
        "raw_fsl_result_root": BASE / "eval_tsy_image_space/mllm_fsl_subset_align_old_test_many_way/result_origin_flow",
        "rae_fsl_result_root": RAE_BASE / "mllm_fsl_subset_agent_lot_way_91",
    },
}


# =============================================================================
# 2. Final/reference parameters
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


# =============================================================================
# 3. OAT sensitivity grids
#
# IMPORTANT:
# Each group changes only ONE hyperparameter; all remaining parameters use
# REFERENCE. Every grid includes the reference value.
# =============================================================================

GRIDS: Dict[str, List[float]] = {
    # Stomach
    "stomach_alpha":    [0.05, 0.15, 0.25, 0.35, 0.45],
    "stomach_guard":    [0.005, 0.01, 0.02, 0.03, 0.05],
    "stomach_gap_thr":  [0.01, 0.03, 0.05, 0.07, 0.10],

    # Esophagus
    "esophagus_top1_thr": [0.10, 0.15, 0.20, 0.25, 0.30],
    "esophagus_fsl_thr":  [0.03, 0.05, 0.10, 0.15, 0.20],
    "esophagus_gap_thr":  [0.005, 0.01, 0.02, 0.03, 0.05],

    # Duodenum
    "duodenum_top1_thr": [0.70, 0.75, 0.80, 0.85, 0.90],
    "duodenum_fsl_thr":  [0.01, 0.02, 0.03, 0.05, 0.08],
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


# =============================================================================
# 4. Helpers
# =============================================================================

def fmt_value(x: float) -> str:
    # Filesystem-safe compact value.
    return f"{x:g}".replace("-", "m").replace(".", "p")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def validate_sweep_configuration() -> None:
    """Fail early if a sweep is inconsistent with the final parameters."""
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
        if not any(abs(float(value) - reference_value) < 1e-12
                   for value in GRIDS[param]):
            raise ValueError(
                f"Sweep grid {param!r} does not include its reference value "
                f"{reference_value:g}: {GRIDS[param]}"
            )


def build_command(
    setting_name: str,
    params: Dict[str, float],
    output_root: Path,
) -> List[str]:
    s = SETTINGS[setting_name]

    return [
        sys.executable,
        "-u",
        str(V2_PY),
        str(s["input_root"]),
        "--output_root", str(output_root),

        "--raw_fsl_result_root", str(s["raw_fsl_result_root"]),
        "--raw_fsl_distance_field", "distances_sinkhorn_w2_image_mapped",
        "--raw_fsl_support_agg", "min",

        "--rae_fsl_result_root", str(s["rae_fsl_result_root"]),

        "--stomach_alpha", str(params["stomach_alpha"]),
        "--stomach_guard", str(params["stomach_guard"]),
        "--stomach_gap_thr", str(params["stomach_gap_thr"]),

        "--esophagus_top1_thr", str(params["esophagus_top1_thr"]),
        "--esophagus_fsl_thr", str(params["esophagus_fsl_thr"]),
        "--esophagus_gap_thr", str(params["esophagus_gap_thr"]),

        "--duodenum_top1_thr", str(params["duodenum_top1_thr"]),
        "--duodenum_fsl_thr", str(params["duodenum_fsl_thr"]),

        "--overwrite",
    ]


def summarize_one_run(
    *,
    setting: str,
    sweep_param: str,
    sweep_value: float,
    output_root: Path,
    params: Dict[str, float],
) -> tuple[dict, List[dict]]:
    summary_csv = output_root / "batch_fusion_summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing summary: {summary_csv}")

    rows = load_csv(summary_csv)

    # Paper sensitivity: latent RAE branch only.
    latent_rows = [
        r for r in rows
        if str(r.get("space", "")).strip().lower() == "latent"
        and str(r.get("fsl_mode", "")).strip().lower() == "traj"
    ]

    if not latent_rows:
        raise RuntimeError(f"No latent/traj rows in {summary_csv}")

    expected_organ = PARAM_TO_ORGAN[sweep_param]
    organ_rows = [r for r in latent_rows if r.get("organ") == expected_organ]

    fusion_all = [100.0 * to_float(r["fusion_acc"]) for r in latent_rows]
    fsl_all = [100.0 * to_float(r["fsl_acc"]) for r in latent_rows]
    mllm_all = [100.0 * to_float(r["mllm_acc"]) for r in latent_rows]

    fusion_organ = [100.0 * to_float(r["fusion_acc"]) for r in organ_rows]
    fsl_organ = [100.0 * to_float(r["fsl_acc"]) for r in organ_rows]

    fixed = sum(to_int(r.get("fixed_fsl_errors")) for r in latent_rows)
    broken = sum(to_int(r.get("broken_fsl_correct")) for r in latent_rows)
    guard = sum(to_int(r.get("guard_used")) for r in latent_rows)
    override = sum(to_int(r.get("override_used")) for r in latent_rows)

    macro_row = {
        "setting": setting,
        "sweep_param": sweep_param,
        "sweep_organ": expected_organ,
        "sweep_value": sweep_value,
        "is_reference_value": int(abs(sweep_value - REFERENCE[sweep_param]) < 1e-12),

        "macro_fusion_acc_percent": mean(fusion_all),
        "macro_fsl_acc_percent": mean(fsl_all),
        "macro_mllm_acc_percent": mean(mllm_all),
        "macro_fusion_minus_fsl_pp": mean(fusion_all) - mean(fsl_all),

        "organ_macro_fusion_acc_percent": mean(fusion_organ),
        "organ_macro_fsl_acc_percent": mean(fsl_organ),
        "organ_macro_fusion_minus_fsl_pp": mean(fusion_organ) - mean(fsl_organ),

        "fixed_fsl_errors_total": fixed,
        "broken_fsl_correct_total": broken,
        "net_fixed_total": fixed - broken,
        "guard_used_total": guard,
        "override_used_total": override,

        "n_datasets": len(latent_rows),
        "n_organ_datasets": len(organ_rows),
        "output_root": str(output_root),
    }

    # Store complete parameter vector for reproducibility.
    for k, v in params.items():
        macro_row[k] = v

    dataset_rows: List[dict] = []
    for r in latent_rows:
        dataset_rows.append({
            "setting": setting,
            "sweep_param": sweep_param,
            "sweep_organ": expected_organ,
            "sweep_value": sweep_value,
            "is_reference_value": macro_row["is_reference_value"],
            "city": r.get("city"),
            "organ": r.get("organ"),
            "total": r.get("total"),
            "fsl_acc_percent": 100.0 * to_float(r.get("fsl_acc")),
            "mllm_acc_percent": 100.0 * to_float(r.get("mllm_acc")),
            "fusion_acc_percent": 100.0 * to_float(r.get("fusion_acc")),
            "fusion_minus_fsl_correct": to_int(r.get("fusion_minus_fsl_correct")),
            "fixed_fsl_errors": to_int(r.get("fixed_fsl_errors")),
            "broken_fsl_correct": to_int(r.get("broken_fsl_correct")),
            "guard_used": to_int(r.get("guard_used")),
            "override_used": to_int(r.get("override_used")),
            "output_root": str(output_root),
        })

    return macro_row, dataset_rows


# =============================================================================
# 5. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--settings",
        nargs="+",
        choices=sorted(SETTINGS),
        default=["few_way", "lot_way"],
        help="Which external settings to evaluate.",
    )
    p.add_argument(
        "--params",
        nargs="+",
        choices=sorted(GRIDS),
        default=list(GRIDS),
        help="Which hyperparameters to sweep.",
    )
    p.add_argument(
        "--sweep_root",
        type=Path,
        default=SWEEP_ROOT,
    )
    p.add_argument(
        "--reuse_existing",
        action="store_true",
        help="If batch_fusion_summary.csv already exists, do not rerun V2.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print jobs only.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    validate_sweep_configuration()
    sweep_root = args.sweep_root.expanduser().resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    if not V2_PY.is_file():
        raise FileNotFoundError(V2_PY)

    for setting in args.settings:
        for key, path in SETTINGS[setting].items():
            if not path.exists():
                raise FileNotFoundError(f"{setting}.{key}: {path}")

    macro_rows: List[dict] = []
    dataset_rows: List[dict] = []
    run_rows: List[dict] = []

    jobs = []
    for setting in args.settings:
        for param in args.params:
            for value in GRIDS[param]:
                params = dict(REFERENCE)
                params[param] = float(value)

                run_name = f"{param}__{fmt_value(value)}"
                output_root = sweep_root / setting / run_name
                jobs.append((setting, param, float(value), params, output_root))

    print("=" * 110)
    print("[V2 Fusion Hyperparameter Sensitivity / OAT]")
    print(f"V2 PY       = {V2_PY}")
    print(f"settings    = {args.settings}")
    print(f"parameters  = {args.params}")
    print(f"jobs        = {len(jobs)}")
    print(f"sweep root  = {sweep_root}")
    print("=" * 110)

    for job_idx, (setting, param, value, params, output_root) in enumerate(jobs, 1):
        summary_csv = output_root / "batch_fusion_summary.csv"

        print(
            f"\n[{job_idx:03d}/{len(jobs):03d}] "
            f"{setting} | {param}={value:g} | "
            f"reference={REFERENCE[param]:g}"
        )

        command = build_command(setting, params, output_root)

        status = "pending"
        error = ""

        if args.dry_run:
            print(" ".join(command))
            status = "dry_run"
        elif args.reuse_existing and summary_csv.is_file():
            print(f"[REUSE] {summary_csv}")
            status = "reused"
        else:
            output_root.mkdir(parents=True, exist_ok=True)

            log_path = output_root / "run.log"
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    command,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            if proc.returncode != 0:
                status = "failed"
                error = f"returncode={proc.returncode}; see {log_path}"
                print(f"[FAILED] {error}")
            else:
                status = "ok"

        run_row = {
            "setting": setting,
            "sweep_param": param,
            "sweep_organ": PARAM_TO_ORGAN[param],
            "sweep_value": value,
            "reference_value": REFERENCE[param],
            "is_reference_value": int(abs(value - REFERENCE[param]) < 1e-12),
            "status": status,
            "error": error,
            "output_root": str(output_root),
        }
        run_rows.append(run_row)

        if status in {"ok", "reused"}:
            macro_row, detail_rows = summarize_one_run(
                setting=setting,
                sweep_param=param,
                sweep_value=value,
                output_root=output_root,
                params=params,
            )
            macro_rows.append(macro_row)
            dataset_rows.extend(detail_rows)

            print(
                "[RESULT] "
                f"Macro Fusion={macro_row['macro_fusion_acc_percent']:.4f}% | "
                f"FSL={macro_row['macro_fsl_acc_percent']:.4f}% | "
                f"Gain={macro_row['macro_fusion_minus_fsl_pp']:+.4f} pp | "
                f"{PARAM_TO_ORGAN[param]} Macro="
                f"{macro_row['organ_macro_fusion_acc_percent']:.4f}% | "
                f"Fixed={macro_row['fixed_fsl_errors_total']} | "
                f"Broken={macro_row['broken_fsl_correct_total']}"
            )

    # -------------------------------------------------------------------------
    # Global outputs
    # -------------------------------------------------------------------------
    run_fields = [
        "setting", "sweep_param", "sweep_organ", "sweep_value",
        "reference_value", "is_reference_value",
        "status", "error", "output_root",
    ]
    runs_csv = sweep_root / "sensitivity_runs.csv"
    write_csv(runs_csv, run_rows, run_fields)

    # Always create these two CSVs, even if there are zero successful runs.
    # This avoids printing paths for files that were never actually written.
    macro_fields = [
        "setting", "sweep_param", "sweep_organ", "sweep_value",
        "is_reference_value",

        "macro_fusion_acc_percent",
        "macro_fsl_acc_percent",
        "macro_mllm_acc_percent",
        "macro_fusion_minus_fsl_pp",

        "organ_macro_fusion_acc_percent",
        "organ_macro_fsl_acc_percent",
        "organ_macro_fusion_minus_fsl_pp",

        "fixed_fsl_errors_total",
        "broken_fsl_correct_total",
        "net_fixed_total",
        "guard_used_total",
        "override_used_total",

        "n_datasets", "n_organ_datasets",

        *REFERENCE.keys(),
        "output_root",
    ]

    detail_fields = [
        "setting", "sweep_param", "sweep_organ", "sweep_value",
        "is_reference_value",
        "city", "organ", "total",
        "fsl_acc_percent", "mllm_acc_percent", "fusion_acc_percent",
        "fusion_minus_fsl_correct",
        "fixed_fsl_errors", "broken_fsl_correct",
        "guard_used", "override_used",
        "output_root",
    ]

    macro_csv = sweep_root / "sensitivity_macro.csv"
    dataset_csv = sweep_root / "sensitivity_by_dataset.csv"

    write_csv(
        macro_csv,
        macro_rows,
        macro_fields,
    )

    write_csv(
        dataset_csv,
        dataset_rows,
        detail_fields,
    )

    # Helpful reference snapshot. This is NOT automatic parameter selection.
    reference_rows = [
        r for r in macro_rows
        if r["is_reference_value"] == 1
    ]

    payload = {
        "reference_parameters": REFERENCE,
        "grids": GRIDS,
        "note": (
            "OAT sensitivity only. Do not select final hyperparameters from "
            "test-set maxima; choose parameters on an independent validation set."
        ),
        "reference_rows": reference_rows,
    }

    reference_json = sweep_root / "sensitivity_best_reference.json"
    with reference_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Strict run-status validation
    # -------------------------------------------------------------------------
    failed_rows = [
        r for r in run_rows
        if r["status"] == "failed"
    ]

    ok_rows = [
        r for r in run_rows
        if r["status"] in {"ok", "reused"}
    ]

    print("\n" + "=" * 110)
    print("[SWEEP SUMMARY]")
    print(f"total jobs = {len(run_rows)}")
    print(f"success    = {len(ok_rows)}")
    print(f"failed     = {len(failed_rows)}")
    print(f"runs       = {runs_csv}")
    print(f"macro      = {macro_csv}")
    print(f"dataset    = {dataset_csv}")
    print(f"reference  = {reference_json}")
    print("=" * 110)

    if failed_rows:
        print("\n[FAILED RUNS]")
        for r in failed_rows:
            print(
                f"  {r['setting']} | "
                f"{r['sweep_param']}={r['sweep_value']} | "
                f"{r['error']}"
            )

        print(
            "\n[ERROR] One or more sensitivity jobs failed. "
            "Check each run's run.log above."
        )

        # Important: non-zero exit status.
        # With shell `set -e`, the outer shell will stop here and will NOT
        # incorrectly print its own [DONE] block.
        raise RuntimeError(
            f"{len(failed_rows)}/{len(run_rows)} sensitivity jobs failed."
        )

    if not macro_rows:
        raise RuntimeError(
            "No successful sensitivity results were summarized."
        )

    print("\n" + "=" * 110)
    print("[DONE]")
    print(f"runs      = {runs_csv}")
    print(f"macro     = {macro_csv}")
    print(f"dataset   = {dataset_csv}")
    print("=" * 110)


if __name__ == "__main__":
    main()
