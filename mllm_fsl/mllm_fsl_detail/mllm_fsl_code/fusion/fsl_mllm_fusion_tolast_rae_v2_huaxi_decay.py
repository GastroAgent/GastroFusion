#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WCH GastroFusion: Adaptive F1-v5 (Decay) geometric scores + GastroMLLM.

Reads per-organ Adaptive F1-v5 query_results.json (results[*].branches.adaptive).
All eight organ-specific fusion values must be supplied explicitly via CLI (normally
by the calling shell script). Fusion decision rules and output formats are preserved.
Legacy raw/image-space FSL inputs and uniform/tolast_mean fallback are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


# =============================================================================
# 第一个代码的区域默认融合策略：保持不变
# =============================================================================

REGION_STRATEGIES = {
    "胃": "gap_guard_fusion",
    "食管": "high_conf_override",
    "十二指肠": "very_high_conf_override",
}

LEGACY_CITY_NAMES = {"苍溪", "自贡", "南充", "华西"}
DEFAULT_HUAXI_CITY = "华西"
FSL_MODES = ("traj", "direct")

# NEW RAE latent-traj protocol. Keep fixed so that latent FSL exactly matches
# the RAE score_summary row:
#   mode=adaptive, score=adaptive_f1_v5, aggregation=softmin
RAE_LATENT_MODE = "adaptive"
RAE_LATENT_SCORE = "adaptive_f1_v5"
RAE_LATENT_SUPPORT_AGG = "softmin"
CACHE_SUFFIX = "_fsl_cache.json"
IGNORED_DIR_NAMES = {
    "备份",
    "backup",
    "backups",
    "bak",
    "__pycache__",
}

IMAGE_PATH_KEYS = (
    "image_path",
    "query_path",
    "img_path",
    "path",
    "image",
)


# =============================================================================
# 基础工具
# =============================================================================

def normalize_class(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace(" ", "").strip().lower()


def class_eq(a: Any, b: Any) -> bool:
    return normalize_class(a) == normalize_class(b)


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


def safe_bool(x: Any, default: bool | None = None) -> bool | None:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        value = x.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    return default


def normalize_path_key(value: Any) -> str:
    if value is None:
        return ""
    return os.path.normpath(str(value)).replace("\\", "/")


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_ignored_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    return any(
        str(part).lower() in IGNORED_DIR_NAMES
        for part in relative.parts
    )


def load_json_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        results = data
        data = {"results": results}
    elif isinstance(data, dict):
        results = data.get("results")
    else:
        raise ValueError(f"JSON 顶层必须是 dict 或 list：{path}")

    if not isinstance(results, list):
        raise ValueError(f'JSON 中缺少有效的 "results" 列表：{path}')

    cleaned: list[dict[str, Any]] = []
    for index, record in enumerate(results):
        if not isinstance(record, dict):
            raise ValueError(
                f"results[{index}] 不是 dict：{path}"
            )
        cleaned.append(record)

    return data, cleaned


# =============================================================================
# FSL view：FSL 只从 *_fsl_cache.json 中读取
# =============================================================================

def ranking_from_class_min(class_min: Any) -> list[dict[str, Any]]:
    if not isinstance(class_min, dict):
        return []

    ranking: list[dict[str, Any]] = []
    for class_name, distance in class_min.items():
        distance_value = safe_float(distance)
        if class_name is None or distance_value is None:
            continue

        ranking.append({
            "class": str(class_name),
            "distance": float(distance_value),
        })

    ranking.sort(key=lambda x: x["distance"])
    return ranking


def calc_gap_from_ranking(ranking: list[dict[str, Any]]) -> float:
    if len(ranking) > 1:
        return float(
            ranking[1]["distance"]
            - ranking[0]["distance"]
        )
    return 999.0


def validate_ranking(
    ranking: Any,
    *,
    mode: str,
    source_path: Path,
    record_index: int,
) -> list[dict[str, Any]]:
    if not isinstance(ranking, list):
        ranking = []

    cleaned: list[dict[str, Any]] = []

    for item in ranking:
        if not isinstance(item, dict):
            continue

        class_name = item.get("class")
        distance = safe_float(item.get("distance"))

        if class_name is None or distance is None:
            continue

        cleaned.append({
            "class": str(class_name),
            "distance": float(distance),
        })

    cleaned.sort(key=lambda x: x["distance"])

    if not cleaned:
        raise ValueError(
            f"{mode} 缺少有效 ranking："
            f"file={source_path}, record={record_index}"
        )

    return cleaned


# =============================================================================
# NEW RAE latent-traj FSL
# Fixed protocol: Adaptive F1-v5 query-level ranking
# =============================================================================
def _rae_raw_query_key(record: dict[str, Any]) -> tuple[str, str]:
    for field in (
        "target_image",
        "x0_path",
        "query_path",
        "image_path",
    ):
        value = record.get(field)
        if value:
            return ("path", normalize_path_key(value))

    for field in (
        "data_id",
        "question_id",
        "query_idx",
        "idx",
        "index",
    ):
        value = record.get(field)
        if value is not None:
            return (field, str(value))

    raise KeyError(
        "RAE result 记录无法确定 query："
        "缺少 target_image/x0_path/query_path/image_path/"
        "data_id/question_id 等字段。"
    )


def _rae_candidate_class(record: dict[str, Any]) -> str:
    for field in (
        "matched_class",
        "label_B",
        "candidate_class",
        "support_class",
    ):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)

    raise KeyError(
        "RAE result 记录缺少候选类别字段 "
        "(matched_class / label_B / candidate_class / support_class)"
    )


def _load_rae_result_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        records = data["results"]
    else:
        raise ValueError(
            "RAE result.json 顶层必须是 list 或包含 results 列表的 dict："
            f"{path}"
        )

    out: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"RAE result[{i}] 不是 dict：{path}"
            )
        out.append(record)

    return out


def resolve_rae_latent_huaxi_source(
    *,
    result_root: Path,
    organ: str,
) -> tuple[Path, str]:
    """
    华西目录按 organ 组织：

        result_root/
          胃/
            **/query_results.json
          食管/
          十二指肠/

    只读取 Adaptive F1-v5 query_results.json。
    """
    organ_root = (
        result_root / str(organ)
    ).expanduser().resolve()

    if not organ_root.is_dir():
        raise FileNotFoundError(
            "找不到华西 NEW RAE organ 目录：\n"
            f"  result_root={result_root}\n"
            f"  organ={organ}\n"
            f"  expected_dir={organ_root}"
        )

    query_candidates = sorted(
        p.resolve()
        for p in organ_root.rglob("query_results.json")
        if p.is_file()
        and not any(
            str(part).lower() in IGNORED_DIR_NAMES
            for part in p.relative_to(organ_root).parts
        )
    )

    if len(query_candidates) > 1:
        listing = "\n".join(
            f"    - {p}" for p in query_candidates
        )
        raise RuntimeError(
            "同一华西 organ 下找到多个 query_results.json，"
            "无法自动确定使用哪个：\n"
            f"  organ={organ}\n{listing}"
        )

    if len(query_candidates) == 1:
        return query_candidates[0], "query_results"

    raise FileNotFoundError(
        f"Adaptive F1-v5 requires query_results.json under {organ_root}; "
        "legacy pair-level result.json does not contain adaptive rankings."
    )


def _adapt_decay_branches_to_scores_block_huaxi(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require the trained F1-v5 adaptive branch, never fall back to uniform."""
    adapted = []
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError(f"F1-v5 results[{i}] must be a dict")
        branches = item.get("branches")
        if not isinstance(branches, dict) or not isinstance(branches.get("adaptive"), dict):
            raise KeyError(f"F1-v5 results[{i}].branches.adaptive missing")
        new_item = dict(item)
        new_item["scores"] = {
            RAE_LATENT_MODE: {RAE_LATENT_SUPPORT_AGG: {RAE_LATENT_SCORE: branches["adaptive"]}}
        }
        adapted.append(new_item)
    return adapted


def build_rae_latent_views_from_query_results(
    query_results_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    """
    直接读取：
        results[*]["branches"]["adaptive"]
    """
    with query_results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(f"Adaptive F1-v5 output must contain results list: {query_results_path}")
    queries = _adapt_decay_branches_to_scores_block_huaxi(data["results"])
    data = dict(data)
    data["queries"] = queries

    if not isinstance(data.get("queries"), list):
        raise ValueError(
            "RAE query_results.json 必须包含 queries 列表："
            f"{query_results_path}"
        )

    views: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    basename_candidates: dict[
        str,
        list[tuple[str, str]],
    ] = {}

    for i, query in enumerate(data["queries"]):
        if not isinstance(query, dict):
            raise TypeError(
                f"queries[{i}] 不是 dict：{query_results_path}"
            )

        query_path = (
            query.get("query_path")
            or query.get("target_image")
            or query.get("image_path")
        )
        if not query_path:
            raise KeyError(
                "RAE query_results query 缺少 query_path/"
                "target_image/image_path："
                f"file={query_results_path}, query={i}"
            )

        scores = query.get("scores")
        try:
            block = (
                scores[RAE_LATENT_MODE]
                [RAE_LATENT_SUPPORT_AGG]
                [RAE_LATENT_SCORE]
            )
        except Exception as exc:
            raise KeyError(
                "RAE query_results 缺少固定结果："
                f"scores[{RAE_LATENT_MODE!r}]"
                f"[{RAE_LATENT_SUPPORT_AGG!r}]"
                f"[{RAE_LATENT_SCORE!r}] | "
                f"file={query_results_path}, query={i}"
            ) from exc

        if not isinstance(block, dict):
            raise TypeError(
                "RAE query_results 目标 score block 不是 dict："
                f"file={query_results_path}, query={i}"
            )

        ranking = validate_ranking(
            block.get("ranking") or [],
            mode="Adaptive F1-v5",
            source_path=query_results_path,
            record_index=i,
        )

        top1 = (
            block.get("pred")
            or block.get("top1")
            or ranking[0]["class"]
        )
        top1 = str(top1)

        gap = safe_float(block.get("gap"))
        if gap is None:
            gap = calc_gap_from_ranking(ranking)

        matches = [
            float(item["distance"])
            for item in ranking
            if class_eq(item.get("class"), top1)
        ]
        distance = (
            matches[0]
            if matches
            else float(ranking[0]["distance"])
        )

        key = (
            "path",
            normalize_path_key(query_path),
        )

        views[key] = {
            "top1": top1,
            "gap": float(gap),
            "distance": float(distance),
            "ranking": ranking,
            "source_field": (
                "rae_query_results:"
                f"{RAE_LATENT_MODE}/"
                f"{RAE_LATENT_SUPPORT_AGG}/"
                f"{RAE_LATENT_SCORE}"
            ),
            "mode": "traj",
            "support_agg": RAE_LATENT_SUPPORT_AGG,
            "ground_truth": query.get("ground_truth"),
        }

        basename = os.path.basename(key[1])
        basename_candidates.setdefault(
            basename,
            [],
        ).append(key)

    basename_index: dict[
        str,
        tuple[str, str],
    ] = {}

    for basename, keys in basename_candidates.items():
        if len(keys) == 1:
            basename_index[basename] = keys[0]

    return views, basename_index


def build_rae_latent_views(*, source_path: Path, source_kind: str):
    if source_kind != "query_results":
        raise ValueError("Adaptive F1-v5 only supports query_results.json")
    return build_rae_latent_views_from_query_results(source_path)


def get_rae_fsl_view_for_record(
    *,
    rae_views: dict[
        tuple[str, str],
        dict[str, Any],
    ],
    basename_index: dict[
        str,
        tuple[str, str],
    ],
    main_record: dict[str, Any],
    cache_record: dict[str, Any],
    record_index: int,
    source_path: Path,
) -> dict[str, Any]:
    """
    将 MLLM/cache query 对齐到 NEW RAE query view。

    优先：
      1) full path
      2) id
      3) 唯一 basename
    """
    candidate_records = (
        main_record,
        cache_record,
    )

    for record in candidate_records:
        for field in (
            *IMAGE_PATH_KEYS,
            "target_image",
            "x0_path",
        ):
            value = record.get(field)
            if not value:
                continue

            key = (
                "path",
                normalize_path_key(value),
            )
            if key in rae_views:
                return rae_views[key]

    for record in candidate_records:
        for field in (
            "data_id",
            "idx",
            "index",
            "query_idx",
            "question_id",
        ):
            value = record.get(field)
            if value is None:
                continue

            for raw_field in (
                "data_id",
                "idx",
                "index",
                "query_idx",
                "question_id",
            ):
                key = (
                    raw_field,
                    str(value),
                )
                if key in rae_views:
                    return rae_views[key]

    for record in candidate_records:
        for field in (
            *IMAGE_PATH_KEYS,
            "target_image",
            "x0_path",
        ):
            value = record.get(field)
            if not value:
                continue

            basename = os.path.basename(
                normalize_path_key(value)
            )
            key = basename_index.get(
                basename
            )

            if (
                key is not None
                and key in rae_views
            ):
                return rae_views[key]

    raise KeyError(
        "无法把 Huaxi MLLM/cache query 对齐到 NEW RAE："
        f"record={record_index}, source={source_path}"
    )


def get_fsl_block_from_cache(
    cache_record: dict[str, Any],
    *,
    source_path: Path,
    record_index: int,
) -> tuple[dict[str, Any], str]:
    """
    FSL 结果严格从 cache_record 中读取。

    兼容：
      - fsl_full_result
      - fsl_result
    """
    for field_name in (
        "fsl_full_result",
        "fsl_result",
    ):
        value = cache_record.get(field_name)

        if isinstance(value, dict) and value:
            return value, field_name

    raise KeyError(
        "FSL cache 记录中找不到有效的 "
        "`fsl_full_result` 或 `fsl_result`："
        f"file={source_path}, record={record_index}"
    )


def get_fsl_view_from_cache(
    cache_record: dict[str, Any],
    fsl_mode: str,
    *,
    source_path: Path,
    record_index: int,
) -> dict[str, Any]:
    """
    从 cache 文件中提取 Traj / Direct FSL view。

    traj:
      ranking / top1 / gap / distance

    direct:
      ranking_direct / top1_direct / gap_direct / distance_direct

    同时兼容部分旧 cache 中单独的 fsl_full_result_direct。
    """
    fsl, source_field = get_fsl_block_from_cache(
        cache_record,
        source_path=source_path,
        record_index=record_index,
    )

    if fsl_mode == "traj":
        ranking = (
            fsl.get("ranking")
            or cache_record.get("fsl_ranking")
            or []
        )

        if not ranking:
            ranking = ranking_from_class_min(
                fsl.get("class_min")
                or fsl.get("class_min_distances")
                or fsl.get("class_distances")
                or {}
            )

        ranking = validate_ranking(
            ranking,
            mode="Traj",
            source_path=source_path,
            record_index=record_index,
        )

        top1 = (
            fsl.get("top1")
            or cache_record.get("fsl_top1")
            or ranking[0]["class"]
        )
        top1 = str(top1)

        gap = safe_float(fsl.get("gap"))
        if gap is None:
            gap = calc_gap_from_ranking(ranking)

        distance = safe_float(fsl.get("distance"))
        if distance is None:
            matches = [
                item["distance"]
                for item in ranking
                if class_eq(item["class"], top1)
            ]
            if not matches:
                raise ValueError(
                    f"Traj top1={top1!r} 不在 ranking 中："
                    f"file={source_path}, record={record_index}"
                )
            distance = float(matches[0])

        return {
            "top1": top1,
            "gap": float(gap),
            "distance": float(distance),
            "ranking": ranking,
            "source_field": source_field,
            "mode": "traj",
        }

    if fsl_mode == "direct":
        ranking = fsl.get("ranking_direct") or []

        direct_block = cache_record.get(
            "fsl_full_result_direct",
            {},
        ) or {}

        if not ranking:
            ranking = (
                direct_block.get("ranking")
                or direct_block.get("ranking_direct")
                or []
            )

        if not ranking:
            ranking = ranking_from_class_min(
                fsl.get("class_min_direct")
                or fsl.get("class_min_distances_direct")
                or fsl.get("class_distances_direct")
                or direct_block.get("class_min")
                or direct_block.get("class_min_direct")
                or direct_block.get("class_min_distances_direct")
                or direct_block.get("class_distances_direct")
                or {}
            )

        ranking = validate_ranking(
            ranking,
            mode="Direct",
            source_path=source_path,
            record_index=record_index,
        )

        top1 = (
            fsl.get("top1_direct")
            or direct_block.get("top1")
            or direct_block.get("top1_direct")
            or ranking[0]["class"]
        )
        top1 = str(top1)

        gap_raw = (
            fsl.get("gap_direct")
            if fsl.get("gap_direct") is not None
            else direct_block.get("gap")
        )
        gap = safe_float(gap_raw)

        if gap is None:
            gap = calc_gap_from_ranking(ranking)

        distance_raw = (
            fsl.get("distance_direct")
            if fsl.get("distance_direct") is not None
            else direct_block.get("distance")
        )
        distance = safe_float(distance_raw)

        if distance is None:
            matches = [
                item["distance"]
                for item in ranking
                if class_eq(item["class"], top1)
            ]
            if not matches:
                raise ValueError(
                    f"Direct top1={top1!r} 不在 ranking 中："
                    f"file={source_path}, record={record_index}"
                )
            distance = float(matches[0])

        return {
            "top1": top1,
            "gap": float(gap),
            "distance": float(distance),
            "ranking": ranking,
            "source_field": source_field,
            "mode": "direct",
        }

    raise ValueError(f"Unknown fsl_mode: {fsl_mode}")


# =============================================================================
# MLLM：严格从同名前缀的非 cache JSON 中读取
# =============================================================================

def get_mllm_correct_from_main(
    main_record: dict[str, Any],
    *,
    source_path: Path,
    record_index: int,
) -> bool:
    value = safe_bool(
        main_record.get("mllm_correct")
    )

    if value is None:
        raise KeyError(
            "非 cache JSON 记录中缺少有效的 `mllm_correct`："
            f"file={source_path}, record={record_index}"
        )

    return value


def get_mllm_score_map_from_main(
    main_record: dict[str, Any],
    *,
    source_path: Path,
    record_index: int,
) -> dict[str, Any]:
    """
    只从非 cache JSON 中读取 mllm_score_map。

    若字段缺失或为空，保持原第一个代码的保守语义：
      - gap_guard_fusion 中无法获得 MLLM score 时退回 FSL；
      - override 类策略中无有效 score 时退回 FSL。
    """
    score_map = main_record.get("mllm_score_map")

    if score_map is None:
        print(
            f"[WARN] mllm_score_map missing -> fallback to FSL | "
            f"file={source_path}, record={record_index}"
        )
        return {}

    if not isinstance(score_map, dict):
        raise TypeError(
            "`mllm_score_map` 不是 dict："
            f"file={source_path}, record={record_index}, "
            f"type={type(score_map).__name__}"
        )

    if not score_map:
        print(
            f"[WARN] empty mllm_score_map -> fallback to FSL | "
            f"file={source_path}, record={record_index}"
        )

    return score_map


# =============================================================================
# 第一个代码的融合策略
# 仅把输入从“单一混合 record”改成 FSL view + MLLM score map
# =============================================================================

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
        ranking[i]["class"]:
        exp_values[i] / total
        for i in range(len(ranking))
    }


def _get_mllm_entry_exact(
    mllm_score_map: dict[str, Any],
    class_name: str,
) -> dict[str, Any]:
    """
    保持第一个代码的类别 score 读取逻辑：优先精确 key。
    为避免 main/cache 类别仅空格或大小写不同导致完全取不到，
    精确 key 不存在时再做 normalize_class 兜底。
    """
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


def predict_gap_guard_fusion(
    *,
    fsl_view: dict[str, Any],
    mllm_score_map: dict[str, Any],
    alpha: float,
    guard: float,
    gap_thr: float,
) -> tuple[str, dict[str, Any]]:
    """
    保留原第一个代码：
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

    # 原代码当 key 不存在时给 1.0；当整个 score_map 为空时这里显式回退 FSL。
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

    fsl_scores = softmax_from_distances(
        ranking
    )

    best_class = fsl_top1
    best_score = -1.0
    combined_scores: dict[str, float] = {}

    for item in ranking:
        class_name = item["class"]

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
    保留原第一个代码的 high_conf_override / very_high_conf_override。
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


def fusion_predict(
    *,
    fsl_view: dict[str, Any],
    mllm_score_map: dict[str, Any],
    region: str,
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
        f"Unknown strategy: {strategy}"
    )


def build_region_params(region: str, args: argparse.Namespace) -> dict[str, Any]:
    """All numeric parameters originate from the calling shell/CLI."""
    params: dict[str, Any] = {"strategy": REGION_STRATEGIES[region]}
    if region == "胃":
        params.update(alpha=args.stomach_alpha, guard=args.stomach_guard, gap_thr=args.stomach_gap_thr)
    elif region == "食管":
        params.update(top1_thr=args.esophagus_top1_thr, fsl_thr=args.esophagus_fsl_thr, gap_thr=args.esophagus_gap_thr)
    elif region == "十二指肠":
        params.update(top1_thr=args.duodenum_top1_thr, fsl_thr=args.duodenum_fsl_thr, gap_thr=None)
    return params


# =============================================================================
# 两个 JSON 的记录对齐：照第二个代码
# =============================================================================

def record_image_key(
    record: dict[str, Any],
) -> str:
    for key in IMAGE_PATH_KEYS:
        value = record.get(key)

        if value:
            return normalize_path_key(value)

    return ""


def record_idx_key(
    record: dict[str, Any],
) -> str:
    value = record.get("idx")

    if value is None:
        value = record.get("index")

    if value is None:
        value = record.get("query_idx")

    return (
        ""
        if value is None
        else str(value)
    )


def record_ground_truth(
    record: dict[str, Any],
) -> Any:
    for key in (
        "ground_truth",
        "gt",
        "label",
        "true_label",
    ):
        if record.get(key) is not None:
            return record.get(key)

    return None


def build_unique_index(
    records: list[dict[str, Any]],
    key_func,
) -> dict[str, int] | None:
    index: dict[str, int] = {}

    for i, record in enumerate(records):
        key = key_func(record)

        if not key or key in index:
            return None

        index[key] = i

    return index


def validate_aligned_ground_truth(
    pairs: Iterable[
        tuple[int, dict[str, Any], dict[str, Any]]
    ],
    *,
    cache_path: Path,
    main_path: Path,
) -> None:
    for index, cache_record, main_record in pairs:
        cache_gt = record_ground_truth(
            cache_record
        )
        main_gt = record_ground_truth(
            main_record
        )

        if (
            cache_gt is not None
            and main_gt is not None
            and not class_eq(cache_gt, main_gt)
        ):
            raise ValueError(
                "cache 与非 cache 记录 GT 不一致："
                f"record={index}, "
                f"cache_gt={cache_gt!r}, "
                f"main_gt={main_gt!r}, "
                f"cache={cache_path}, "
                f"main={main_path}"
            )


def align_record_pairs(
    cache_records: list[dict[str, Any]],
    main_records: list[dict[str, Any]],
    *,
    cache_path: Path,
    main_path: Path,
) -> tuple[
    list[
        tuple[
            int,
            dict[str, Any],
            dict[str, Any],
        ]
    ],
    str,
]:
    """
    对齐优先级：
      1. image_path / query_path 等；
      2. idx / index / query_idx；
      3. 长度一致时按 position，并逐条检查 GT。
    """
    if not cache_records:
        raise ValueError(
            f"FSL cache 的 results 为空：{cache_path}"
        )

    if not main_records:
        raise ValueError(
            f"非 cache JSON 的 results 为空：{main_path}"
        )

    cache_image_index = build_unique_index(
        cache_records,
        record_image_key,
    )
    main_image_index = build_unique_index(
        main_records,
        record_image_key,
    )

    if (
        cache_image_index is not None
        and main_image_index is not None
        and set(cache_image_index)
        == set(main_image_index)
    ):
        pairs = []

        for main_i, main_record in enumerate(
            main_records
        ):
            key = record_image_key(main_record)
            cache_record = cache_records[
                cache_image_index[key]
            ]
            pairs.append(
                (
                    main_i,
                    cache_record,
                    main_record,
                )
            )

        validate_aligned_ground_truth(
            pairs,
            cache_path=cache_path,
            main_path=main_path,
        )

        return pairs, "image_path"

    cache_idx_index = build_unique_index(
        cache_records,
        record_idx_key,
    )
    main_idx_index = build_unique_index(
        main_records,
        record_idx_key,
    )

    if (
        cache_idx_index is not None
        and main_idx_index is not None
        and set(cache_idx_index)
        == set(main_idx_index)
    ):
        pairs = []

        for main_i, main_record in enumerate(
            main_records
        ):
            key = record_idx_key(main_record)
            cache_record = cache_records[
                cache_idx_index[key]
            ]
            pairs.append(
                (
                    main_i,
                    cache_record,
                    main_record,
                )
            )

        validate_aligned_ground_truth(
            pairs,
            cache_path=cache_path,
            main_path=main_path,
        )

        return pairs, "idx"

    if len(cache_records) != len(main_records):
        raise ValueError(
            "无法对齐 cache 和非 cache JSON："
            f"cache_records={len(cache_records)}, "
            f"main_records={len(main_records)}, "
            f"cache={cache_path}, "
            f"main={main_path}"
        )

    pairs = [
        (
            i,
            cache_records[i],
            main_records[i],
        )
        for i in range(len(main_records))
    ]

    validate_aligned_ground_truth(
        pairs,
        cache_path=cache_path,
        main_path=main_path,
    )

    return pairs, "position"


# =============================================================================
# 自动 city / organ / space / region
# =============================================================================

def _infer_organ_from_filename(
    cache_path: Path,
) -> str:
    name = cache_path.name

    match = re.search(
        r"_(?:image|latent)_(.+?)_fsl_cache\.json$",
        name,
        flags=re.IGNORECASE,
    )

    return (
        match.group(1)
        if match
        else ""
    )


def infer_region_from_path(
    path: Path,
    input_root: Optional[Path] = None,
) -> Optional[str]:
    try:
        relative = (
            path.relative_to(input_root)
            if input_root is not None
            else path
        )
    except ValueError:
        relative = path

    for part in reversed(relative.parts):
        text = str(part).strip()

        if text in (
            "胃",
            "食管",
            "十二指肠",
        ):
            return text

        if text.lower() == "kvasir":
            return "kvasir"

    text = str(relative)

    if "十二指肠" in text:
        return "十二指肠"

    if "食管" in text:
        return "食管"

    if "胃" in text:
        return "胃"

    if "kvasir" in text.lower():
        return "kvasir"

    return None


def infer_metadata(
    *,
    input_root: Path,
    cache_path: Path,
) -> dict[str, str]:
    """
    自动兼容：
      外院: root/城市/器官/*.json
      华西: root/器官/*.json
      Kvasir: 路径中包含 kvasir
    """
    relative = cache_path.relative_to(
        input_root
    )

    parent_parts = relative.parent.parts

    region = (
        infer_region_from_path(
            cache_path,
            input_root,
        )
        or ""
    )

    city = ""
    organ = ""

    if region == "kvasir":
        city = "Kvasir"
        organ = "kvasir"

    elif (
        len(parent_parts) >= 2
        and parent_parts[0]
        in LEGACY_CITY_NAMES
    ):
        city = parent_parts[0]
        organ = parent_parts[1]

    elif len(parent_parts) >= 1:
        city = DEFAULT_HUAXI_CITY
        organ = parent_parts[0]

    else:
        city = DEFAULT_HUAXI_CITY
        organ = _infer_organ_from_filename(
            cache_path
        )

    name_lower = cache_path.name.lower()

    if "_image_" in name_lower:
        space = "image"
    elif "_latent_" in name_lower:
        space = "latent"
    else:
        space = "unknown"

    return {
        "city": city,
        "organ": organ,
        "space": space,
        "region": region,
    }


# =============================================================================
# 单个 cache/main 文件对 + 单个 mode
# =============================================================================

def process_pair_mode(
    *,
    input_root: Path,
    cache_path: Path,
    main_path: Path,
    output_path: Path,
    fsl_mode: str,
    params: dict[str, Any],
    region: str,
    verbose_details: bool,
    rae_fsl_result_root: Path | None = None,
) -> dict[str, Any]:
    cache_data, cache_records = (
        load_json_results(cache_path)
    )
    main_data, main_records = (
        load_json_results(main_path)
    )

    aligned_pairs, alignment_method = (
        align_record_pairs(
            cache_records,
            main_records,
            cache_path=cache_path,
            main_path=main_path,
        )
    )

    main_only_rae_mode = (
        cache_path.resolve() == main_path.resolve()
    )
    if main_only_rae_mode:
        alignment_method = "mllm_main_direct_to_rae"

    metadata = infer_metadata(
        input_root=input_root,
        cache_path=cache_path,
    )

    # NEW RAE latent-traj source.
    # Other combinations keep the original cache behavior.
    if main_only_rae_mode and not (
        metadata["space"] == "latent"
        and fsl_mode == "traj"
    ):
        raise ValueError(
            "main-only MLLM input is supported only for latent+traj, because "
            "FSL must come from --rae_fsl_result_root."
        )
    rae_source_path: Path | None = None
    rae_source_kind: str | None = None
    rae_views: dict[
        tuple[str, str],
        dict[str, Any],
    ] | None = None
    rae_basename_index: dict[
        str,
        tuple[str, str],
    ] | None = None

    if (
        metadata["space"] == "latent"
        and fsl_mode == "traj"
    ):
        if rae_fsl_result_root is None:
            raise ValueError(
                "latent+traj 必须提供 --rae_fsl_result_root"
            )

        (
            rae_source_path,
            rae_source_kind,
        ) = resolve_rae_latent_huaxi_source(
            result_root=rae_fsl_result_root,
            organ=metadata["organ"],
        )

        (
            rae_views,
            rae_basename_index,
        ) = build_rae_latent_views(
            source_path=rae_source_path,
            source_kind=rae_source_kind,
        )

        print(
            "[RAE FSL] Huaxi latent/traj | "
            f"organ={metadata['organ']} | "
            f"{RAE_LATENT_MODE}/"
            f"{RAE_LATENT_SUPPORT_AGG}/"
            f"{RAE_LATENT_SCORE} | "
            f"source={rae_source_kind} | "
            f"queries={len(rae_views)} | "
            f"file={rae_source_path}"
        )

    output_records: list[
        dict[str, Any]
    ] = []

    fsl_correct_count = 0
    mllm_correct_count = 0
    fusion_correct_count = 0

    fixed_count = 0
    broken_count = 0
    guard_count = 0
    override_count = 0

    fsl_source_fields: Counter[str] = (
        Counter()
    )

    for (
        record_index,
        cache_record,
        main_record,
    ) in aligned_pairs:

        ground_truth = (
            record_ground_truth(main_record)
            if record_ground_truth(
                main_record
            ) is not None
            else record_ground_truth(
                cache_record
            )
        )

        if ground_truth is None:
            raise KeyError(
                "cache 与非 cache 记录都缺少 "
                "ground_truth："
                f"cache={cache_path}, "
                f"main={main_path}, "
                f"record={record_index}"
            )

        if (
            rae_views is not None
            and rae_basename_index is not None
        ):
            fsl_view = get_rae_fsl_view_for_record(
                rae_views=rae_views,
                basename_index=rae_basename_index,
                main_record=main_record,
                cache_record=cache_record,
                record_index=record_index,
                source_path=rae_source_path,
            )
        else:
            fsl_view = get_fsl_view_from_cache(
                cache_record,
                fsl_mode,
                source_path=cache_path,
                record_index=record_index,
            )

        mllm_correct = (
            get_mllm_correct_from_main(
                main_record,
                source_path=main_path,
                record_index=record_index,
            )
        )

        mllm_score_map = (
            get_mllm_score_map_from_main(
                main_record,
                source_path=main_path,
                record_index=record_index,
            )
        )

        fusion_pred, fusion_diag = (
            fusion_predict(
                fsl_view=fsl_view,
                mllm_score_map=mllm_score_map,
                region=region,
                params=params,
            )
        )

        fsl_correct = class_eq(
            fsl_view["top1"],
            ground_truth,
        )

        fusion_correct = class_eq(
            fusion_pred,
            ground_truth,
        )

        fsl_correct_count += int(
            fsl_correct
        )
        mllm_correct_count += int(
            mllm_correct
        )
        fusion_correct_count += int(
            fusion_correct
        )

        guard_count += int(
            bool(
                fusion_diag.get(
                    "used_guard",
                    False,
                )
            )
        )

        override_count += int(
            bool(
                fusion_diag.get(
                    "used_override",
                    False,
                )
            )
        )

        fsl_source_fields[
            fsl_view["source_field"]
        ] += 1

        if (
            not fsl_correct
            and fusion_correct
        ):
            fixed_count += 1

        elif (
            fsl_correct
            and not fusion_correct
        ):
            broken_count += 1

        merged_record = dict(
            main_record
        )

        merged_record.update({
            "ground_truth":
                ground_truth,

            "fusion_v2_region":
                region,

            "fusion_v2_strategy":
                params["strategy"],

            "fusion_v2_params":
                params,

            "fusion_v2_fsl_mode":
                fsl_mode,

            "fusion_v2_fsl_pred":
                fsl_view["top1"],

            "fusion_v2_fsl_gap":
                fsl_view["gap"],

            "fusion_v2_fsl_distance":
                fsl_view["distance"],

            "fusion_v2_fsl_ranking":
                fsl_view["ranking"],

            "fusion_v2_fsl_correct":
                fsl_correct,

            "fusion_v2_mllm_correct":
                mllm_correct,

            "fusion_v2_pred":
                fusion_pred,

            "fusion_v2_correct":
                fusion_correct,

            "fusion_v2_diagnostics":
                fusion_diag,

            "fusion_v2_fsl_source_json":
                str(
                    rae_source_path
                    if rae_source_path is not None
                    else cache_path
                ),

            "fusion_v2_fsl_source_field":
                fsl_view["source_field"],

            "fusion_v2_rae_source_kind":
                rae_source_kind,

            "fusion_v2_rae_mode":
                (
                    RAE_LATENT_MODE
                    if rae_source_path is not None
                    else None
                ),

            "fusion_v2_rae_score":
                (
                    RAE_LATENT_SCORE
                    if rae_source_path is not None
                    else None
                ),

            "fusion_v2_rae_support_agg":
                (
                    RAE_LATENT_SUPPORT_AGG
                    if rae_source_path is not None
                    else None
                ),

            "fusion_v2_mllm_source_json":
                str(main_path),
        })

        output_records.append(
            merged_record
        )

    total = len(output_records)

    if total <= 0:
        raise ValueError(
            f"没有可输出记录："
            f"cache={cache_path}, "
            f"main={main_path}"
        )

    summary = {
        **metadata,
        "fsl_mode":
            fsl_mode,
        "strategy":
            params["strategy"],
        "total":
            total,

        "fsl_correct":
            fsl_correct_count,
        "fsl_acc":
            fsl_correct_count / total,

        "mllm_correct":
            mllm_correct_count,
        "mllm_acc":
            mllm_correct_count / total,

        "fusion_correct":
            fusion_correct_count,
        "fusion_acc":
            fusion_correct_count / total,

        "fusion_minus_fsl_correct":
            fusion_correct_count
            - fsl_correct_count,

        "fusion_minus_mllm_correct":
            fusion_correct_count
            - mllm_correct_count,

        "fixed_fsl_errors":
            fixed_count,

        "broken_fsl_correct":
            broken_count,

        "net_fixed":
            fixed_count
            - broken_count,

        "guard_used":
            guard_count,

        "override_used":
            override_count,

        "alignment_method":
            alignment_method,

        "fsl_source_field_counts":
            dict(fsl_source_fields),

        "alpha":
            params.get("alpha"),

        "guard":
            params.get("guard"),

        "top1_thr":
            params.get("top1_thr"),

        "fsl_thr":
            params.get("fsl_thr"),

        "gap_thr":
            params.get("gap_thr"),

        "fsl_cache_json":
            (
                None
                if main_only_rae_mode
                else str(cache_path)
            ),

        "rae_fsl_source_json":
            (
                str(rae_source_path)
                if rae_source_path is not None
                else None
            ),

        "rae_fsl_source_kind":
            rae_source_kind,

        "rae_fsl_mode":
            (
                RAE_LATENT_MODE
                if rae_source_path is not None
                else None
            ),

        "rae_fsl_score":
            (
                RAE_LATENT_SCORE
                if rae_source_path is not None
                else None
            ),

        "rae_fsl_support_agg":
            (
                RAE_LATENT_SUPPORT_AGG
                if rae_source_path is not None
                else None
            ),

        "mllm_json":
            str(main_path),

        "output_json":
            str(output_path),
    }

    # 以非 cache JSON 为主文件，FSL 信息来自 cache
    output_data = {
        "config": {
            "region":
                region,
            "fsl_mode":
                fsl_mode,
            "strategy":
                params["strategy"],
            "params":
                params,
            "fsl_source_policy":
                (
                    "Adaptive F1-v5"
                    if rae_source_path is not None
                    else "only *_fsl_cache.json"
                ),
            "mllm_source_policy":
                (
                    "main-only MLLM JSON"
                    if main_only_rae_mode
                    else "only paired non-cache JSON"
                ),
        },

        "sources": {
            "fsl_cache_json":
                (
                    None
                    if main_only_rae_mode
                    else str(cache_path)
                ),
            "rae_fsl_source_json":
                (
                    str(rae_source_path)
                    if rae_source_path is not None
                    else None
                ),
            "mllm_json":
                str(main_path),
            "cache_top_level_keys":
                (
                    []
                    if main_only_rae_mode
                    else sorted(
                        str(key)
                        for key
                        in cache_data.keys()
                    )
                ),
            "main_top_level_keys":
                sorted(
                    str(key)
                    for key
                    in main_data.keys()
                ),
            "alignment_method":
                alignment_method,
        },

        "summary":
            summary,

        "results":
            output_records,
    }

    atomic_write_json(
        output_path,
        output_data,
    )

    print(
        f"[DONE]"
        f"[{metadata['city']}/"
        f"{metadata['organ']}/"
        f"{metadata['space']}]"
        f"[{fsl_mode}] "
        f"strategy={params['strategy']} | "
        f"FSL={fsl_correct_count}/{total} "
        f"({fsl_correct_count / total:.4f}) | "
        f"MLLM={mllm_correct_count}/{total} "
        f"({mllm_correct_count / total:.4f}) | "
        f"Fusion={fusion_correct_count}/{total} "
        f"({fusion_correct_count / total:.4f}) | "
        f"delta={fusion_correct_count - fsl_correct_count:+d}"
    )

    if verbose_details:
        print(
            f"       fixed={fixed_count}, "
            f"broken={broken_count}, "
            f"net={fixed_count - broken_count}, "
            f"guard={guard_count}, "
            f"override={override_count}, "
            f"align={alignment_method}"
        )

    return summary


# =============================================================================
# Paired batch helpers：照第二个代码
# =============================================================================

def collect_file_pairs(
    *,
    input_root: Path,
    output_root: Path,
) -> tuple[
    list[tuple[Path, Path]],
    list[dict[str, str]],
]:
    """
    Collect MLLM inputs for fusion.

    Preferred legacy mode:
      *_fsl_cache.json + paired non-cache JSON.

    Current Huaxi NEW-RAE mode:
      the new MLLM batch runner produces ONLY non-cache main JSON files,
      while FSL is read independently from --rae_fsl_result_root.
      In that case no cache anchor is needed. We return (main, main) as a
      lightweight compatibility pair; process_pair_mode will still obtain
      the actual FSL view exclusively from offline NEW RAE.
    """
    pairs: list[tuple[Path, Path]] = []
    missing: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # 1) Legacy paired mode: keep backward compatibility.
    # ------------------------------------------------------------------
    for cache_path in sorted(input_root.rglob(f"*{CACHE_SUFFIX}")):
        if not cache_path.is_file():
            continue
        if is_ignored_path(cache_path, input_root):
            continue
        if path_is_inside(cache_path, output_root):
            continue

        base_name = cache_path.name[:-len(CACHE_SUFFIX)] + ".json"
        main_path = cache_path.with_name(base_name)

        if not main_path.is_file():
            missing.append({
                "fsl_cache_json": str(cache_path),
                "expected_mllm_json": str(main_path),
            })
            continue

        pairs.append((cache_path, main_path))

    if pairs:
        return pairs, missing

    # ------------------------------------------------------------------
    # 2) NEW RAE main-only mode.
    #    Scan the new MLLM batch-runner outputs directly.
    # ------------------------------------------------------------------
    main_only: list[tuple[Path, Path]] = []

    for main_path in sorted(input_root.rglob("*.json")):
        if not main_path.is_file():
            continue
        if main_path.name.endswith(CACHE_SUFFIX):
            continue
        if is_ignored_path(main_path, input_root):
            continue
        if path_is_inside(main_path, output_root):
            continue

        # Skip fusion/batch bookkeeping JSONs if input_root is reused.
        name_lower = main_path.name.lower()
        if (
            "fusion_v2" in name_lower
            or name_lower.startswith("batch_")
            or name_lower.startswith("missing_")
        ):
            continue

        try:
            data, records = load_json_results(main_path)
        except Exception:
            continue

        # A valid new MLLM main JSON must contain at least one record with
        # mllm_score_map. This avoids accidentally treating unrelated JSON as
        # fusion input.
        if not records:
            continue
        if not any(
            isinstance(r.get("mllm_score_map"), dict)
            and bool(r.get("mllm_score_map"))
            for r in records
        ):
            continue

        # (main, main) is intentional. For latent+traj the true FSL source is
        # NEW RAE, so the first element is only a compatibility anchor for the
        # existing batching/metadata code.
        main_only.append((main_path, main_path))

    if main_only:
        print(
            f"[INFO] No *_fsl_cache.json pairs found; "
            f"using NEW-RAE main-only mode with {len(main_only)} MLLM JSON files."
        )
        return main_only, []

    return [], missing


def make_output_path(
    *,
    input_root: Path,
    output_root: Path,
    cache_path: Path,
    fsl_mode: str,
) -> Path:
    relative_parent = (
        cache_path.parent.relative_to(
            input_root
        )
    )

    if cache_path.name.endswith(CACHE_SUFFIX):
        base_stem = cache_path.name[:-len(CACHE_SUFFIX)]
    else:
        # NEW RAE main-only mode: cache_path is actually the MLLM main JSON.
        base_stem = cache_path.stem

    return (
        output_root
        / relative_parent
        / (
            f"{base_stem}"
            f"_fusion_v2_{fsl_mode}.json"
        )
    )


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "city",
        "organ",
        "space",
        "region",
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
        "alignment_method",
        "alpha",
        "guard",
        "top1_thr",
        "fsl_thr",
        "gap_thr",
        "fsl_cache_json",
        "rae_fsl_source_json",
        "rae_fsl_source_kind",
        "rae_fsl_mode",
        "rae_fsl_score",
        "rae_fsl_support_agg",
        "mllm_json",
        "output_json",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "第一个代码的分区域融合策略 + "
            "第二个代码的 paired batch / auto-layout 处理逻辑。"
        )
    )

    parser.add_argument(
        "input_root",
        help=(
            "输入根目录。优先支持 legacy *_fsl_cache.json 配对；"
            "若无 cache，则直接读取新 MLLM batch runner 生成的 main JSON。"
        ),
    )

    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "输出根目录。默认："
            "<input_root>/fusion_v2_paired"
        ),
    )

    parser.add_argument(
        "--fsl_mode",
        choices=[
            "traj",
            "direct",
            "both",
        ],
        default="traj",
        help=(
            "traj=仅 trajectory（默认）；"
            "direct=仅 direct；"
            "both=两者都跑（需显式指定，direct 为旧 cache 分支）。"
        ),
    )

    parser.add_argument(
        "--region",
        default=None,
        choices=[
            "胃",
            "食管",
            "十二指肠",
        ],
        help=(
            "可选。默认按每个文件路径自动识别；"
            "指定后所有文件强制使用该 region 策略。"
        ),
    )

    # V2 organ-specific parameters: explicit per organ
    parser.add_argument(
        "--stomach_alpha",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--stomach_guard",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--stomach_gap_thr",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--esophagus_top1_thr",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--esophagus_fsl_thr",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--esophagus_gap_thr",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--duodenum_top1_thr",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--duodenum_fsl_thr",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--rae_fsl_result_root",
        required=True,
        help=(
            "华西 NEW RAE result 根目录。结构为 "
            "<root>/<organ>/**/query_results.json；"
            "只读取 Adaptive F1-v5 query_results.json。"
        ),
    )

    parser.add_argument(
        "--space",
        choices=["all", "image", "latent"],
        default="latent",
        help=(
            "限制处理空间。当前 RAE latent-traj 建议使用 --space latent。"
        ),
    )

    parser.add_argument(
        "--verbose_details",
        action="store_true",
    )

    parser.add_argument(
        "--fail_on_missing_pair",
        action="store_true",
        help=(
            "发现 cache 缺少对应非 cache JSON 时直接失败。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "覆盖已有输出；默认跳过。"
        ),
    )

    args = parser.parse_args()

    input_root = (
        Path(args.input_root)
        .expanduser()
        .resolve()
    )

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"输入根目录不存在："
            f"{input_root}"
        )

    prob_args = {
        "--stomach_alpha": args.stomach_alpha,
        "--esophagus_top1_thr": args.esophagus_top1_thr,
        "--esophagus_fsl_thr": args.esophagus_fsl_thr,
        "--duodenum_top1_thr": args.duodenum_top1_thr,
        "--duodenum_fsl_thr": args.duodenum_fsl_thr,
    }
    for name, value in prob_args.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} 必须在 [0,1]：{value}"
            )

    if args.stomach_guard < 0.0:
        raise ValueError(
            "--stomach_guard 必须非负"
        )
    if args.stomach_gap_thr < 0.0:
        raise ValueError(
            "--stomach_gap_thr 必须非负"
        )
    if args.esophagus_gap_thr < 0.0:
        raise ValueError(
            "--esophagus_gap_thr 必须非负"
        )

    rae_fsl_result_root = (
        Path(args.rae_fsl_result_root)
        .expanduser()
        .resolve()
    )
    if not rae_fsl_result_root.is_dir():
        raise NotADirectoryError(
            "--rae_fsl_result_root 不存在："
            f"{rae_fsl_result_root}"
        )

    output_root = (
        Path(args.output_root)
        .expanduser()
        .resolve()
        if args.output_root
        else (
            input_root
            / "fusion_v2_paired"
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    modes = (
        list(FSL_MODES)
        if args.fsl_mode == "both"
        else [args.fsl_mode]
    )

    pairs, missing_pairs = (
        collect_file_pairs(
            input_root=input_root,
            output_root=output_root,
        )
    )

    # Optional space filter. For the current Huaxi RAE experiment,
    # --space latent makes the run contain only latent files.
    if args.space != "all":
        pairs = [
            (cache_path, main_path)
            for cache_path, main_path in pairs
            if infer_metadata(
                input_root=input_root,
                cache_path=cache_path,
            )["space"] == args.space
        ]

    if missing_pairs:
        print(
            "\n[WARN] 以下 FSL cache "
            "缺少对应非 cache JSON："
        )

        for item in missing_pairs:
            print(
                f"  cache   : "
                f"{item['fsl_cache_json']}\n"
                f"  expected: "
                f"{item['expected_mllm_json']}"
            )

        atomic_write_json(
            output_root
            / "missing_file_pairs.json",
            missing_pairs,
        )

        if args.fail_on_missing_pair:
            raise FileNotFoundError(
                f"发现 "
                f"{len(missing_pairs)} "
                f"个缺失文件对。"
            )

    if not pairs:
        raise RuntimeError(
            f"没有找到有效文件对："
            f"{input_root}"
        )

    detected_meta = [
        infer_metadata(
            input_root=input_root,
            cache_path=cache_path,
        )
        for cache_path, _
        in pairs
    ]

    detected_cities = sorted({
        item["city"]
        for item in detected_meta
        if item["city"]
    })

    detected_organs = sorted({
        item["organ"]
        for item in detected_meta
        if item["organ"]
    })

    detected_spaces = sorted({
        item["space"]
        for item in detected_meta
        if item["space"]
    })

    print("=" * 100)
    print(
        "[BATCH "
        "FSL + MLLM FUSION V2]"
    )
    print(
        f"input_root   : "
        f"{input_root}"
    )
    print(
        f"output_root  : "
        f"{output_root}"
    )
    print(
        f"cities       : "
        f"{', '.join(detected_cities) or 'NA'}"
    )
    print(
        f"organs       : "
        f"{', '.join(detected_organs) or 'NA'}"
    )
    print(
        f"spaces       : "
        f"{', '.join(detected_spaces) or 'NA'}"
    )
    print(
        f"file pairs   : "
        f"{len(pairs)}"
    )
    print(
        f"modes        : "
        f"{', '.join(modes)}"
    )
    print(
        f"region       : "
        f"{args.region or 'AUTO FROM PATH'}"
    )
    print("=" * 100)

    summaries: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    skipped_outputs = 0

    total_jobs = (
        len(pairs)
        * len(modes)
    )

    job_index = 0

    for cache_path, main_path in pairs:
        metadata = infer_metadata(
            input_root=input_root,
            cache_path=cache_path,
        )

        region = (
            args.region
            or metadata["region"]
        )

        if region not in REGION_STRATEGIES:
            for fsl_mode in modes:
                job_index += 1
                failures.append({
                    "fsl_cache_json":
                        str(cache_path),
                    "mllm_json":
                        str(main_path),
                    "fsl_mode":
                        fsl_mode,
                    "error":
                        (
                            "Cannot infer region "
                            "from path"
                        ),
                })

                print(
                    f"[ERROR] 无法识别 region："
                    f"{cache_path}"
                )
            continue

        params = build_region_params(
            region,
            args,
        )

        for fsl_mode in modes:
            job_index += 1

            output_path = make_output_path(
                input_root=input_root,
                output_root=output_root,
                cache_path=cache_path,
                fsl_mode=fsl_mode,
            )

            print(
                f"\n[{job_index}/"
                f"{total_jobs}] "
                f"{cache_path.relative_to(input_root)} "
                f"| city={metadata['city']} "
                f"| organ={metadata['organ']} "
                f"| space={metadata['space']} "
                f"| region={region} "
                f"| mode={fsl_mode} "
                f"| strategy="
                f"{params['strategy']}"
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                skipped_outputs += 1

                print(
                    f"[SKIP] 输出已存在："
                    f"{output_path}\n"
                    "       需要覆盖时添加 "
                    "--overwrite"
                )

                continue

            try:
                summary = (
                    process_pair_mode(
                        input_root=input_root,
                        cache_path=cache_path,
                        main_path=main_path,
                        output_path=output_path,
                        fsl_mode=fsl_mode,
                        params=params,
                        region=region,
                        verbose_details=(
                            args.verbose_details
                        ),
                        rae_fsl_result_root=(
                            rae_fsl_result_root
                        ),
                    )
                )

                summaries.append(
                    summary
                )

            except Exception as exc:
                failure = {
                    "fsl_cache_json":
                        str(cache_path),
                    "mllm_json":
                        str(main_path),
                    "city":
                        metadata["city"],
                    "organ":
                        metadata["organ"],
                    "space":
                        metadata["space"],
                    "region":
                        region,
                    "fsl_mode":
                        fsl_mode,
                    "error":
                        repr(exc),
                }

                failures.append(
                    failure
                )

                print(
                    f"[ERROR] "
                    f"mode={fsl_mode}, "
                    f"cache={cache_path}\n"
                    f"        {repr(exc)}"
                )

    summary_json_path = (
        output_root
        / "batch_fusion_v2_summary.json"
    )

    summary_csv_path = (
        output_root
        / "batch_fusion_v2_summary.csv"
    )

    failure_path = (
        output_root
        / "batch_fusion_v2_failures.json"
    )

    batch_summary = {
        "input_root":
            str(input_root),

        "output_root":
            str(output_root),

        "cities":
            detected_cities,

        "organs":
            detected_organs,

        "spaces":
            detected_spaces,

        "modes":
            modes,

        "region_override":
            args.region,

        "v2_organ_params": {
            "胃": build_region_params(
                "胃",
                args,
            ),
            "食管": build_region_params(
                "食管",
                args,
            ),
            "十二指肠": build_region_params(
                "十二指肠",
                args,
            ),
        },

        "space_filter":
            args.space,

        "rae_fsl_result_root":
            str(rae_fsl_result_root),

        "rae_latent_protocol": {
            "mode": RAE_LATENT_MODE,
            "score": RAE_LATENT_SCORE,
            "support_aggregation":
                RAE_LATENT_SUPPORT_AGG,
        },

        "discovered_file_pairs":
            len(pairs),

        "expected_jobs":
            total_jobs,

        "successful_jobs_this_run":
            len(summaries),

        "skipped_existing_outputs":
            skipped_outputs,

        "failed_jobs":
            len(failures),

        "missing_file_pairs":
            len(missing_pairs),

        "results":
            summaries,

        "failures":
            failures,
    }

    atomic_write_json(
        summary_json_path,
        batch_summary,
    )

    write_summary_csv(
        summary_csv_path,
        summaries,
    )

    atomic_write_json(
        failure_path,
        failures,
    )

    print("\n" + "=" * 100)
    print("[BATCH DONE]")
    print(
        f"file pairs       : "
        f"{len(pairs)}"
    )
    print(
        f"expected jobs    : "
        f"{total_jobs}"
    )
    print(
        f"successful       : "
        f"{len(summaries)}"
    )
    print(
        f"skipped existing : "
        f"{skipped_outputs}"
    )
    print(
        f"failed           : "
        f"{len(failures)}"
    )
    print(
        f"missing pairs    : "
        f"{len(missing_pairs)}"
    )
    print(
        f"summary JSON     : "
        f"{summary_json_path}"
    )
    print(
        f"summary CSV      : "
        f"{summary_csv_path}"
    )
    print(
        f"failures JSON    : "
        f"{failure_path}"
    )
    print("=" * 100)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
