#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import subprocess
import json
from pathlib import Path


# ============================================================
# Path
# ============================================================

PY = (
    "/mnt/inaisfs/data/home/tansy_criait/"
    "wass_flow_match_胃/mllm_fsl_result/"
    "fsl_mllm_fusion_tolast_rae_v2_huaxi.py"
)


INPUT_ROOT = (
    "/mnt/inaisfs/data/home/tansy_criait/"
    "wass_flow_match_胃/mllm_fsl_result/"
    "mllm_code/mllm_result_huaxi_val"
)


RAE_ROOT = (
    "/mnt/inaisfs/data/home/tansy_criait/"
    "qsz/qcjw/our_method/批量测试/"
    "result_831_val/offline_align_rae_metric_huaxi_new"
)


OUTPUT_ROOT = Path(
    "/mnt/inaisfs/data/home/tansy_criait/"
    "wass_flow_match_胃/mllm_fsl_result/"
    "val_grid_search_results"
)


# ============================================================
# Grid Search Space
# ============================================================

GRID = {

    # stomach
    "stomach_alpha": [
        0.25,
        0.35,
        0.50,
        0.75
    ],

    "stomach_guard": [
        0.01,
        0.03,
        0.05
    ],

    "stomach_gap_thr": [
        0.01,
        0.02,
        0.05
    ],


    # esophagus
    "esophagus_top1_thr": [
        0.20,
        0.30,
        0.40
    ],

    "esophagus_fsl_thr": [
        0.10,
        0.15,
        0.20
    ],

    "esophagus_gap_thr": [
        0.01,
        0.02,
        0.05
    ],


    # duodenum
    "duodenum_top1_thr": [
        0.80,
        0.90,
        0.95
    ],

    "duodenum_fsl_thr": [
        0.03,
        0.05,
        0.10
    ]
}



# ============================================================
# Run one experiment
# ============================================================

def run_one(config, idx):

    out_dir = OUTPUT_ROOT / f"exp_{idx:04d}"

    cmd = [
        "python",
        "-u",
        PY,

        INPUT_ROOT,

        "--output_root",
        str(out_dir),

        "--fsl_mode",
        "traj",

        "--space",
        "latent",

        "--rae_fsl_result_root",
        RAE_ROOT,

        "--overwrite",

    ]


    for k, v in config.items():

        cmd.extend(
            [
                f"--{k}",
                str(v)
            ]
        )


    print("=" * 100)
    print(f"RUN {idx}")
    print(config)
    print("=" * 100)


    subprocess.run(
        cmd,
        check=True
    )


    summary_json = (
        out_dir /
        "batch_fusion_v2_summary.json"
    )


    if not summary_json.exists():

        raise RuntimeError(
            f"Missing summary: {summary_json}"
        )


    with open(summary_json, "r") as f:

        data = json.load(f)


    return calculate_score(
        config,
        data
    )



# ============================================================
# New scoring
# ============================================================

def calculate_score(config, data):

    results = data["results"]


    fsl_accs = []
    fusion_accs = []


    rescue = 0
    degradation = 0


    total = 0

    fusion_correct = 0
    fsl_correct = 0


    for item in results:


        fsl_accs.append(
            item["fsl_acc"]
        )


        fusion_accs.append(
            item["fusion_acc"]
        )


        rescue += max(
            0,
            item["fusion_correct"]
            -
            item["fsl_correct"]
        )


        degradation += max(
            0,
            item["fsl_correct"]
            -
            item["fusion_correct"]
        )


        total += item["total"]


        fusion_correct += item["fusion_correct"]

        fsl_correct += item["fsl_correct"]



    macro_fsl = (
        sum(fsl_accs)
        /
        len(fsl_accs)
    )


    macro_fusion = (
        sum(fusion_accs)
        /
        len(fusion_accs)
    )


    net_rescue = rescue - degradation



    # ========================================================
    # Objective
    #
    # 1. Fusion cannot hurt FSL
    # 2. Reward rescue
    # ========================================================

    if macro_fusion < macro_fsl:

        score = -999

    else:

        rescue_gain = (
            net_rescue / total
        )

        score = (
            macro_fusion
            +
            0.1 * rescue_gain
        )


    return {

        **config,


        # ranking
        "score": score,


        # accuracy
        "macro_fusion_acc_percent":
            macro_fusion * 100,


        "macro_fsl_acc_percent":
            macro_fsl * 100,


        # count
        "fusion_correct":
            fusion_correct,


        "fsl_correct":
            fsl_correct,


        "rescue":
            rescue,


        "degradation":
            degradation,


        "net_rescue":
            net_rescue,


        "valid":
            macro_fusion >= macro_fsl,


    }



# ============================================================
# Main
# ============================================================

def main():


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    keys = list(
        GRID.keys()
    )


    values = [
        GRID[k]
        for k in keys
    ]


    all_results = []


    combinations = list(
        itertools.product(*values)
    )


    print(
        f"Total grid experiments: {len(combinations)}"
    )


    for idx, combo in enumerate(combinations):


        config = dict(
            zip(
                keys,
                combo
            )
        )


        result = run_one(
            config,
            idx
        )


        all_results.append(
            result
        )


        print(
            "[RESULT]",
            result
        )



    # best first

    all_results.sort(
        key=lambda x:x["score"],
        reverse=True
    )


    with open(
        OUTPUT_ROOT /
        "grid_search_all.json",

        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )



    best = all_results[0]


    with open(
        OUTPUT_ROOT /
        "best_config.json",

        "w"
    ) as f:

        json.dump(
            best,
            f,
            indent=2,
            ensure_ascii=False
        )


    print("\n")
    print("="*100)
    print("BEST CONFIG")
    print(best)
    print("="*100)



if __name__ == "__main__":

    main()