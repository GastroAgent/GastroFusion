#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成对批处理 Trajectory FSL + MLLM 融合结果（V2 器官特异融合）。

本版本在保留 NEW RAE latent-traj FSL 来源的基础上，
将原统一 alpha/guard 融合改为 V2 的器官特异策略：

  胃：
      gap_guard_fusion
  食管：
      high_conf_override
  十二指肠：
      very_high_conf_override

所有八个融合数值参数必须由命令行指定；Python 不设置数值默认值。


目录结构示例：
    INPUT_ROOT/
      苍溪/
        十二指肠/
          Med-qwen3-vl-RL_image_十二指肠_fsl_cache.json
          Med-qwen3-vl-RL_image_十二指肠.json
          Med-qwen3-vl-RL_latent_十二指肠_fsl_cache.json
          Med-qwen3-vl-RL_latent_十二指肠.json
      自贡/
      南充/

数据来源约束：
    1. FSL 的 Traj 结果：
       - image + traj：从独立 Flow result.json 读取
         `distances_sinkhorn_w2_image_mapped`（trajectory_score_key=tolast_image_feat）。
       - latent + traj：从新的 RAE offline 结果读取，固定使用：
           mode        = uniform
           score       = tolast_mean
           aggregation = min
         优先读取 query_results.json；若不存在则回退到 result.json 重建相同 ranking。
       - *_fsl_cache.json 只继续作为原批处理遍历/MLLM 文件配对与 query 对齐锚点，
         latent-traj 不再从 cache 内部读取 FSL ranking/top1。

    2. MLLM 信息：
       只从与 cache 文件同目录、同名前缀的非 cache JSON 读取。
       其中：
         - mllm_correct 用于统计 MLLM baseline；
         - mllm_score_map 用于 FSL + MLLM Softmax 融合。

    3. 本版本只处理：
         - fsl_mode=traj

默认只遍历：
    苍溪、自贡、南充

默认输出目录：
    <input_root>/fusion_direct_traj/

运行：
    python fsl_mllm_fusion_paired_batch.py \
      /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/result_other_0729

可选：
    --output_root /path/to/output

    # 胃
    --stomach_alpha 0.35
    --stomach_guard 0.001
    --stomach_gap_thr 1.0

    # 食管
    --esophagus_top1_thr 0.30
    --esophagus_fsl_thr 0.15
    --esophagus_gap_thr 0.02

    # 十二指肠
    --duodenum_top1_thr 0.90
    --duodenum_fsl_thr 0.05

    --verbose_details
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CITY_ORDER = ("苍溪", "自贡", "南充")
FSL_MODES = ("traj",)

# =============================================================================
# V2 organ-specific fusion defaults
# =============================================================================
# Fusion strategies are fixed in Python; all numerical thresholds are required
# CLI arguments supplied by the experiment shell script.
REGION_STRATEGIES = {
    "胃": "gap_guard_fusion",
    "食管": "high_conf_override",
    "十二指肠": "very_high_conf_override",
}

CACHE_SUFFIX = "_fsl_cache.json"
IGNORED_DIR_NAMES = {"备份", "backup", "backups", "bak", "__pycache__"}
IMAGE_PATH_KEYS = (
    "image_path",
    "query_path",
    "img_path",
    "path",
    "image",
    "x0",
)


# =============================================================================
# 基础工具
# =============================================================================
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


def normalize_class(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace(" ", "").strip().lower()


def class_eq(a: Any, b: Any) -> bool:
    return normalize_class(a) == normalize_class(b)


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


def record_image_key(record: dict[str, Any]) -> str:
    for key in IMAGE_PATH_KEYS:
        value = record.get(key)
        if value:
            return normalize_path_key(value)
    return ""


def record_idx_key(record: dict[str, Any]) -> str:
    value = record.get("idx")
    if value is None:
        value = record.get("index")
    if value is None:
        value = record.get("query_idx")
    return "" if value is None else str(value)


def record_ground_truth(record: dict[str, Any]) -> Any:
    for key in ("ground_truth", "gt", "label", "true_label"):
        if record.get(key) is not None:
            return record.get(key)
    return None



def _query_path_values(record: dict[str, Any]) -> list[str]:
    """Return all usable query-image path aliases in stable priority order."""
    values: list[str] = []
    seen: set[str] = set()
    for field in IMAGE_PATH_KEYS + ("target_image", "x0_path"):
        value = record.get(field)
        if not value:
            continue
        norm = normalize_path_key(value)
        if norm and norm not in seen:
            seen.add(norm)
            values.append(norm)
    return values


def _class_basename_alias(class_name: Any, path_value: Any) -> str:
    """Class-aware basename key; safe when the same filename exists in multiple classes."""
    cls = normalize_class(class_name)
    path = normalize_path_key(path_value)
    if not cls or not path:
        return ""
    basename = os.path.basename(path).casefold()
    return f"class_basename::{cls}::{basename}"


def _path_tail_alias(path_value: Any, depth: int = 4) -> str:
    """Root-independent suffix key, normally city/organ/class/file for this project."""
    path = normalize_path_key(path_value)
    if not path:
        return ""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return ""
    tail = "/".join(parts[-max(int(depth), 1):]).casefold()
    return f"tail{depth}::{tail}"


def _add_unique_fallback_candidate(
    candidates: dict[str, set[tuple[str, str]]],
    alias: str,
    canonical_key: tuple[str, str],
) -> None:
    if alias:
        candidates.setdefault(alias, set()).add(canonical_key)


def _finalize_unique_fallback_index(
    candidates: dict[str, set[tuple[str, str]]],
) -> dict[str, tuple[str, str]]:
    """Keep only aliases that identify exactly one canonical query."""
    return {
        alias: next(iter(keys))
        for alias, keys in candidates.items()
        if len(keys) == 1
    }

def is_ignored_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    return any(
        part.lower() in IGNORED_DIR_NAMES
        for part in relative.parts
    )


def path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False



# =============================================================================
# Raw result.json -> FSL Traj(image) ranking
# =============================================================================

RAW_TRAJ_IMAGE_DISTANCE_FIELD = "distances_sinkhorn_w2_image_mapped"

# Adaptive F1-v5 is the only geometric FSL evidence source.
RAE_LATENT_MODE = "adaptive"
RAE_LATENT_SCORE = "adaptive_f1_v5"
RAE_LATENT_SUPPORT_AGG = "softmin"


def get_raw_fsl_view_for_record(
    *,
    raw_views: dict[tuple[str, str], dict[str, Any]],
    basename_index: dict[str, tuple[str, str]],
    main_record: dict[str, Any],
    cache_record: dict[str, Any],
    record_index: int,
    result_path: Path,
) -> dict[str, Any]:
    """
    将 MLLM/cache 的单 query 对齐到 FSL query view。

    匹配优先级：
      1) 强 ID：MLLM idx/data_id/... <-> FSL data_id/question_id/...
      2) 完整路径
      3) ground_truth + basename（解决跨类别同名文件）
      4) 路径末 4 段（city/organ/class/file，忽略不同数据根目录）
      5) 全局唯一 basename
    """
    candidate_records = (main_record, cache_record)

    # 1) Strong query IDs.  Cross-field matching is intentional:
    #    e.g. MLLM idx=3 <-> FSL data_id=3/question_id=3.
    for record in candidate_records:
        for field in ("data_id", "idx", "index", "query_idx", "question_id"):
            value = record.get(field)
            if value is None:
                continue
            value = str(value)
            for raw_field in ("data_id", "question_id", "query_idx", "idx", "index"):
                key = (raw_field, value)
                if key in raw_views:
                    return raw_views[key]

    # 2) Exact full path (works when both pipelines share the same root).
    for record in candidate_records:
        for path_value in _query_path_values(record):
            key = ("path", path_value)
            if key in raw_views:
                return raw_views[key]

    # 3) Class-aware basename.  This is crucial for files such as
    #    胃窦__胃窦__20250711_19_cropped.JPG that occur under multiple classes.
    for record in candidate_records:
        gt = record_ground_truth(record)
        if gt is None:
            continue
        for path_value in _query_path_values(record):
            alias = _class_basename_alias(gt, path_value)
            key = basename_index.get(alias)
            if key is not None and key in raw_views:
                return raw_views[key]

    # 4) Root-independent relative suffix (normally city/organ/class/file).
    for record in candidate_records:
        for path_value in _query_path_values(record):
            alias = _path_tail_alias(path_value, 4)
            key = basename_index.get(alias)
            if key is not None and key in raw_views:
                return raw_views[key]

    # 5) Unique basename only; ambiguous basenames are intentionally absent.
    for record in candidate_records:
        for path_value in _query_path_values(record):
            alias = f"basename::{os.path.basename(path_value).casefold()}"
            key = basename_index.get(alias)
            if key is not None and key in raw_views:
                return raw_views[key]

    debug_ids: dict[str, Any] = {}
    debug_paths: list[str] = []
    for tag, record in (("main", main_record), ("cache", cache_record)):
        for field in ("data_id", "idx", "index", "query_idx", "question_id"):
            if record.get(field) is not None:
                debug_ids[f"{tag}.{field}"] = record.get(field)
        debug_paths.extend(_query_path_values(record))

    raise KeyError(
        "无法把 MLLM/cache query 对齐到 FSL result："
        f"record={record_index}, result={result_path}, "
        f"ids={debug_ids}, ground_truth={record_ground_truth(main_record)!r}, "
        f"paths={debug_paths[:6]}"
    )


def resolve_rae_latent_result_source(
    *,
    result_root: Path,
    city: str,
    organ: str,
) -> tuple[Path, str]:
    """
    Locate the NEW RAE offline FSL output for one city-organ.

    Search under:
        result_root / city / organ / ** /

    Require query_results.json with per-query branches.adaptive ranking.
    Legacy pair-level result.json is intentionally not supported.
    """
    organ_root = (result_root / str(city) / str(organ)).expanduser().resolve()

    if not organ_root.is_dir():
        raise FileNotFoundError(
            "找不到新的 RAE latent FSL city/organ 目录：\n"
            f"  rae_result_root={result_root}\n"
            f"  city={city}\n"
            f"  organ={organ}\n"
            f"  expected_dir={organ_root}"
        )

    query_candidates = sorted(
        p.resolve()
        for p in organ_root.rglob("query_results.json")
        if p.is_file()
        and not any(
            part.lower() in IGNORED_DIR_NAMES
            for part in p.relative_to(organ_root).parts
        )
    )

    if len(query_candidates) > 1:
        listing = "\n".join(f"    - {p}" for p in query_candidates)
        raise RuntimeError(
            "同一个 city/organ 下找到多个 query_results.json，无法自动确定使用哪个：\n"
            f"  city={city}\n"
            f"  organ={organ}\n"
            f"{listing}"
        )

    if len(query_candidates) == 1:
        return query_candidates[0], "query_results"

    raise FileNotFoundError(
        f"Adaptive F1-v5 requires query_results.json under {organ_root}; "
        "legacy result.json has no per-query adaptive ranking."
    )

    result_candidates = sorted(
        p.resolve()
        for p in organ_root.rglob("result.json")
        if p.is_file()
        and not any(
            part.lower() in IGNORED_DIR_NAMES
            for part in p.relative_to(organ_root).parts
        )
    )

    if not result_candidates:
        raise FileNotFoundError(
            "新的 RAE latent FSL 目录下既没有 query_results.json，也没有 result.json：\n"
            f"  city={city}\n"
            f"  organ={organ}\n"
            f"  search_dir={organ_root}"
        )

    if len(result_candidates) > 1:
        listing = "\n".join(f"    - {p}" for p in result_candidates)
        raise RuntimeError(
            "同一个 city/organ 下找到多个 result.json，无法自动确定使用哪个：\n"
            f"  city={city}\n"
            f"  organ={organ}\n"
            f"{listing}"
        )

    return result_candidates[0], "result"


def _adapt_decay_branches_to_scores_block(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read the trained adaptive F1-v5 branch, not the uniform diagnostic branch."""
    adapted = []
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError(f"adaptive query_results results[{i}] must be dict")
        branches = item.get("branches")
        if not isinstance(branches, dict) or not isinstance(branches.get("adaptive"), dict):
            raise KeyError(
                f"query_results results[{i}].branches.adaptive missing; "
                "regenerate FSL output using the adaptive F1-v5 evaluator"
            )
        block = branches["adaptive"]
        new_item = dict(item)
        new_item["scores"] = {RAE_LATENT_MODE: {RAE_LATENT_SUPPORT_AGG: {RAE_LATENT_SCORE: block}}}
        adapted.append(new_item)
    return adapted


def build_rae_latent_views_from_query_results(
    query_results_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    """
    Read the Adaptive F1-v5 branch from query_results.json:

        results[*]["branches"]["adaptive"]

    Besides the canonical path key, build root-independent fallback aliases so
    MLLM JSON under data/数据_merge can align to RAE results produced from
    cross_domain_balanced/... without requiring identical absolute roots.
    """
    with query_results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # === decay-seeds schema 适配 ===
    # 新版 RAE 输出把每条 query 放在顶层 "results" 列表里，且 score 信息在
    # 每个 item 的 "branches[<mode>]" 子结构中（仅有 mode 一层嵌套），而
    # 旧脚本期望 data["queries"] 列表 + item["scores"][mode][agg][score] 三层嵌套。
    # 这里做兼容：如果有 "queries" 就用，没有就退到 "results"。
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        queries = _adapt_decay_branches_to_scores_block(data["results"])
    else:
        raise ValueError(
            "RAE query_results.json 必须包含 queries 或 results 列表："
            f"{query_results_path}"
        )
    data = dict(data)
    data["queries"] = queries

    if not isinstance(data.get("queries"), list):
        raise ValueError(
            "RAE query_results.json 必须包含 queries 列表："
            f"{query_results_path}"
        )

    views: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_candidates: dict[str, set[tuple[str, str]]] = {}

    for i, query in enumerate(data["queries"]):
        if not isinstance(query, dict):
            raise TypeError(
                f"RAE query_results queries[{i}] 不是 dict：{query_results_path}"
            )

        query_path = (
            query.get("query_path")
            or query.get("target_image")
            or query.get("image_path")
            or query.get("x0_path")
            or query.get("x0")
        )
        if not query_path:
            raise KeyError(
                "RAE query_results query 缺少 query_path/target_image/image_path/x0："
                f"file={query_results_path}, query={i}"
            )

        scores = query.get("scores")
        try:
            block = scores[RAE_LATENT_MODE][RAE_LATENT_SUPPORT_AGG][RAE_LATENT_SCORE]
        except Exception as exc:
            raise KeyError(
                "RAE query_results 缺少固定 latent-traj 结果："
                f"scores[{RAE_LATENT_MODE!r}]"
                f"[{RAE_LATENT_SUPPORT_AGG!r}]"
                f"[{RAE_LATENT_SCORE!r}] | "
                f"file={query_results_path}, query={i}"
            ) from exc

        if not isinstance(block, dict):
            raise TypeError(
                "RAE query_results 的目标 score block 不是 dict："
                f"file={query_results_path}, query={i}"
            )

        ranking = validate_ranking(
            block.get("ranking") or [],
            mode="adaptive F1-v5",
            source_path=query_results_path,
            record_index=i,
        )

        top1 = block.get("pred") or block.get("top1") or ranking[0]["class"]
        top1 = str(top1)

        gap = safe_float(block.get("gap"))
        if gap is None:
            gap = calc_gap_from_ranking(ranking)

        matches = [
            float(item["distance"])
            for item in ranking
            if class_eq(item.get("class"), top1)
        ]
        distance = matches[0] if matches else float(ranking[0]["distance"])

        key = ("path", normalize_path_key(query_path))
        gt = query.get("ground_truth") or query.get("gt") or query.get("label")
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
            "ground_truth": gt,
        }

        # If query_results happens to carry IDs, preserve them as canonical aliases
        # by adding fallback tokens is unnecessary; get_raw... will use class/tail.
        _add_unique_fallback_candidate(
            fallback_candidates,
            _class_basename_alias(gt, key[1]),
            key,
        )
        _add_unique_fallback_candidate(
            fallback_candidates,
            _path_tail_alias(key[1], 4),
            key,
        )
        _add_unique_fallback_candidate(
            fallback_candidates,
            f"basename::{os.path.basename(key[1]).casefold()}",
            key,
        )

    fallback_index = _finalize_unique_fallback_index(fallback_candidates)
    return views, fallback_index


def build_rae_latent_views(
    *,
    source_path: Path,
    source_kind: str,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    if source_kind == "query_results":
        return build_rae_latent_views_from_query_results(source_path)

    raise ValueError(f"未知 RAE latent source_kind：{source_kind}")


# =============================================================================
# FSL 读取：严格从 cache 记录读取
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


def get_fsl_block_from_cache(
    cache_record: dict[str, Any],
    *,
    source_path: Path,
    record_index: int,
) -> tuple[dict[str, Any], str]:
    """
    FSL 结果只允许从 cache_record 中读取。
    兼容字段：
      - fsl_full_result
      - fsl_result
    """
    for field_name in ("fsl_full_result", "fsl_result"):
        value = cache_record.get(field_name)
        if isinstance(value, dict) and value:
            return value, field_name

    raise KeyError(
        "FSL cache 记录中找不到有效的 "
        "`fsl_full_result` 或 `fsl_result`："
        f"file={source_path}, record={record_index}"
    )


def calc_gap_from_ranking(ranking: list[dict[str, Any]]) -> float:
    if len(ranking) > 1:
        return float(ranking[1]["distance"] - ranking[0]["distance"])
    return 999.0


def get_fsl_view_from_cache(
    cache_record: dict[str, Any],
    fsl_mode: str,
    *,
    source_path: Path,
    record_index: int,
) -> dict[str, Any]:
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

        top1 = fsl.get("top1") or cache_record.get("fsl_top1")
        if not top1:
            top1 = ranking[0]["class"]
        top1 = str(top1)

        gap = safe_float(fsl.get("gap"))
        if gap is None:
            gap = calc_gap_from_ranking(ranking)

        distance = safe_float(fsl.get("distance"))
        if distance is None:
            matches = [
                x["distance"]
                for x in ranking
                if class_eq(x["class"], top1)
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

        if not ranking:
            ranking = ranking_from_class_min(
                fsl.get("class_min_direct")
                or fsl.get("class_min_distances_direct")
                or fsl.get("class_distances_direct")
                or {}
            )

        ranking = validate_ranking(
            ranking,
            mode="Direct",
            source_path=source_path,
            record_index=record_index,
        )

        top1 = fsl.get("top1_direct")
        if not top1:
            top1 = ranking[0]["class"]
        top1 = str(top1)

        gap = safe_float(fsl.get("gap_direct"))
        if gap is None:
            gap = calc_gap_from_ranking(ranking)

        distance = safe_float(fsl.get("distance_direct"))
        if distance is None:
            matches = [
                x["distance"]
                for x in ranking
                if class_eq(x["class"], top1)
            ]
            if not matches:
                raise ValueError(
                    f"Direct top1={top1!r} 不在 ranking_direct 中："
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

    raise ValueError(f"未知 fsl_mode：{fsl_mode}")


# =============================================================================
# MLLM 读取：严格从非 cache 记录读取
# =============================================================================
def get_mllm_correct_from_main(
    main_record: dict[str, Any],
    *,
    source_path: Path,
    record_index: int,
) -> bool:
    value = safe_bool(main_record.get("mllm_correct"))
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
    从非 cache JSON 中读取 mllm_score_map。

    允许 {}：
    某些 query 的 MLLM scoring_pool 为空，因此没有可用的
    score-level probability。此时返回空字典，后续 fusion_predict()
    会通过 Guard 自动回退到 FSL top1。

    只有字段类型不是 dict 时才认为 JSON 格式异常。
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
            f"[WARN] empty mllm_score_map -> Guard fallback to FSL | "
            f"file={source_path}, record={record_index}"
        )

    return score_map


# =============================================================================
# 两个 JSON 的记录对齐
# =============================================================================
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


def align_record_pairs(
    cache_records: list[dict[str, Any]],
    main_records: list[dict[str, Any]],
    *,
    cache_path: Path,
    main_path: Path,
) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any]]], str]:
    """
    返回：
      [(main_order_index, cache_record, main_record), ...], alignment_method

    优先级：
      1. image_path 等路径字段
      2. idx/index/query_idx
      3. 相同长度下按位置，但逐条校验 GT
    """
    if not cache_records:
        raise ValueError(f"FSL cache 的 results 为空：{cache_path}")
    if not main_records:
        raise ValueError(f"非 cache JSON 的 results 为空：{main_path}")

    cache_image_index = build_unique_index(
        cache_records, record_image_key
    )
    main_image_index = build_unique_index(
        main_records, record_image_key
    )

    if cache_image_index is not None and main_image_index is not None:
        if set(cache_image_index) == set(main_image_index):
            pairs = []
            for main_i, main_record in enumerate(main_records):
                key = record_image_key(main_record)
                cache_record = cache_records[cache_image_index[key]]
                pairs.append((main_i, cache_record, main_record))
            validate_aligned_ground_truth(
                pairs,
                cache_path=cache_path,
                main_path=main_path,
            )
            return pairs, "image_path"

    cache_idx_index = build_unique_index(
        cache_records, record_idx_key
    )
    main_idx_index = build_unique_index(
        main_records, record_idx_key
    )

    if cache_idx_index is not None and main_idx_index is not None:
        if set(cache_idx_index) == set(main_idx_index):
            pairs = []
            for main_i, main_record in enumerate(main_records):
                key = record_idx_key(main_record)
                cache_record = cache_records[cache_idx_index[key]]
                pairs.append((main_i, cache_record, main_record))
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
            f"cache={cache_path}, main={main_path}"
        )

    pairs = [
        (i, cache_records[i], main_records[i])
        for i in range(len(main_records))
    ]
    validate_aligned_ground_truth(
        pairs,
        cache_path=cache_path,
        main_path=main_path,
    )
    return pairs, "position"


def validate_aligned_ground_truth(
    pairs: Iterable[tuple[int, dict[str, Any], dict[str, Any]]],
    *,
    cache_path: Path,
    main_path: Path,
) -> None:
    for index, cache_record, main_record in pairs:
        cache_gt = record_ground_truth(cache_record)
        main_gt = record_ground_truth(main_record)

        if cache_gt is not None and main_gt is not None:
            if not class_eq(cache_gt, main_gt):
                raise ValueError(
                    "cache 与非 cache 记录 GT 不一致："
                    f"record={index}, cache_gt={cache_gt!r}, "
                    f"main_gt={main_gt!r}, "
                    f"cache={cache_path}, main={main_path}"
                )


# =============================================================================
# V2 器官特异融合
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
        ranking[i]["class"]: exp_values[i] / total
        for i in range(len(ranking))
    }


def _get_mllm_entry_exact(
    mllm_score_map: dict[str, Any],
    class_name: str,
) -> dict[str, Any]:
    """
    优先精确类别名；若不存在，再按 normalize_class 兜底。
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
    胃使用的 V2 策略：

      if FSL gap >= gap_thr:
          保留 FSL
      elif MLLM_prob(FSL_top1) < guard:
          保留 FSL
      else:
          alpha * FSL_softmax + (1-alpha) * MLLM_prob
    """
    ranking = fsl_view["ranking"]
    fsl_top1 = fsl_view["top1"]
    gap = float(fsl_view["gap"])

    if not ranking or fsl_top1 == "未知":
        return fsl_top1, {
            "used_guard": True,
            "used_override": False,
            "guard_reason": "invalid_fsl_view",
            "mllm_prob_on_fsl_top1": None,
            "combined_scores": {},
        }

    if not mllm_score_map:
        return fsl_top1, {
            "used_guard": True,
            "used_override": False,
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
            "used_override": False,
            "guard_reason": "fsl_gap",
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
            "combined_scores": {},
        }

    if mllm_fsl_prob < guard:
        return fsl_top1, {
            "used_guard": True,
            "used_override": False,
            "guard_reason": "mllm_prob_on_fsl_top1",
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
            "combined_scores": {},
        }

    fsl_scores = softmax_from_distances(ranking)
    if not fsl_scores:
        return fsl_top1, {
            "used_guard": True,
            "used_override": False,
            "guard_reason": "empty_fsl_softmax",
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
            "combined_scores": {},
        }

    best_class = fsl_top1
    best_score = -float("inf")
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
        "used_override": False,
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
    食管 / 十二指肠使用的 V2 override 策略。

    只有当：
      1) MLLM top1 != FSL top1
      2) MLLM top1 probability > top1_thr
      3) MLLM 对 FSL top1 的 probability < fsl_thr
      4) 若 gap_thr 非 None，则 FSL gap < gap_thr

    才允许 MLLM 覆盖 FSL。
    """
    fsl_top1 = fsl_view["top1"]
    gap = float(fsl_view["gap"])

    if not mllm_score_map:
        return fsl_top1, {
            "used_guard": False,
            "used_override": False,
            "override_reason": "empty_mllm_score_map",
            "mllm_top1_class": None,
            "mllm_top1_prob": None,
            "mllm_prob_on_fsl_top1": None,
            "combined_scores": {},
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
            "used_guard": False,
            "used_override": False,
            "override_reason": "no_valid_mllm_scores",
            "mllm_top1_class": None,
            "mllm_top1_prob": None,
            "mllm_prob_on_fsl_top1": None,
            "combined_scores": {},
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
            "used_guard": False,
            "used_override": True,
            "override_reason": "high_conf_mllm",
            "mllm_top1_class": mllm_top1_class,
            "mllm_top1_prob": mllm_top1_prob,
            "mllm_prob_on_fsl_top1": mllm_fsl_prob,
            "combined_scores": {},
        }

    return fsl_top1, {
        "used_guard": False,
        "used_override": False,
        "override_reason": None,
        "mllm_top1_class": mllm_top1_class,
        "mllm_top1_prob": mllm_top1_prob,
        "mllm_prob_on_fsl_top1": mllm_fsl_prob,
        "combined_scores": {},
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
        diag["region"] = region
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
        diag["region"] = region
        return pred, diag

    raise ValueError(
        f"Unknown strategy for {region}: {strategy}"
    )


def build_region_params(
    region: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    每个器官独立读取自己的 CLI 参数，避免一个全局 alpha/guard
    意外覆盖其他器官。
    """
    if region not in REGION_STRATEGIES:
        raise ValueError(
            f"不支持的器官/region：{region}"
        )

    params = {"strategy": REGION_STRATEGIES[region]}

    if region == "胃":
        params["alpha"] = args.stomach_alpha
        params["guard"] = args.stomach_guard
        params["gap_thr"] = args.stomach_gap_thr

    elif region == "食管":
        params["top1_thr"] = args.esophagus_top1_thr
        params["fsl_thr"] = args.esophagus_fsl_thr
        params["gap_thr"] = args.esophagus_gap_thr

    elif region == "十二指肠":
        params["top1_thr"] = args.duodenum_top1_thr
        params["fsl_thr"] = args.duodenum_fsl_thr
        params["gap_thr"] = None

    return params


# =============================================================================
# 单对文件处理
# =============================================================================
def infer_metadata(
    *,
    input_root: Path,
    cache_path: Path,
) -> dict[str, str]:
    relative = cache_path.relative_to(input_root)

    city = relative.parts[0] if relative.parts else ""
    organ = relative.parts[1] if len(relative.parts) >= 2 else ""

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
    }


def process_pair_mode(
    *,
    input_root: Path,
    cache_path: Path,
    main_path: Path,
    output_path: Path,
    fsl_mode: str,
    region: str,
    params: dict[str, Any],
    verbose_details: bool,
    rae_fsl_result_root: Path | None = None,
) -> dict[str, Any]:
    cache_data, cache_records = load_json_results(cache_path)
    main_data, main_records = load_json_results(main_path)

    aligned_pairs, alignment_method = align_record_pairs(
        cache_records,
        main_records,
        cache_path=cache_path,
        main_path=main_path,
    )

    metadata = infer_metadata(
        input_root=input_root,
        cache_path=cache_path,
    )

    if region != metadata["organ"]:
        raise ValueError(
            "region 与当前文件器官不一致："
            f"region={region}, organ={metadata['organ']}, file={cache_path}"
        )

    # FSL source policy:
    #   image/latent MLLM inputs -> same trained Adaptive F1-v5 FSL results
    #
    # Never use legacy Flow/raw FSL or cached FSL predictions.
    rae_latent_source_path: Path | None = None
    rae_latent_source_kind: str | None = None
    rae_latent_views: dict[tuple[str, str], dict[str, Any]] | None = None
    rae_latent_basename_index: dict[str, tuple[str, str]] | None = None

    if metadata["space"] in ("image", "latent") and fsl_mode == "traj":
        if rae_fsl_result_root is None:
            raise ValueError("Adaptive F1-v5 requires --rae_fsl_result_root")
        rae_latent_source_path, rae_latent_source_kind = resolve_rae_latent_result_source(
            result_root=rae_fsl_result_root,
            city=metadata["city"],
            organ=metadata["organ"],
        )
        rae_latent_views, rae_latent_basename_index = build_rae_latent_views(
            source_path=rae_latent_source_path,
            source_kind=rae_latent_source_kind,
        )
        print(
            f"[ADAPTIVE F1-v5] space={metadata['space']} "
            f"queries={len(rae_latent_views)} file={rae_latent_source_path}"
        )

    output_records: list[dict[str, Any]] = []

    fsl_correct_count = 0
    mllm_correct_count = 0
    fusion_correct_count = 0
    fixed_count = 0
    broken_count = 0
    guard_count = 0
    override_count = 0
    fsl_source_fields: Counter[str] = Counter()

    for record_index, cache_record, main_record in aligned_pairs:
        ground_truth = (
            record_ground_truth(main_record)
            if record_ground_truth(main_record) is not None
            else record_ground_truth(cache_record)
        )
        if ground_truth is None:
            raise KeyError(
                "cache 与非 cache 记录都缺少 ground_truth："
                f"cache={cache_path}, main={main_path}, "
                f"record={record_index}"
            )

        if rae_latent_views is None or rae_latent_basename_index is None:
            raise RuntimeError("Missing Adaptive F1-v5 query-level predictions")
        fsl_view = get_raw_fsl_view_for_record(
            raw_views=rae_latent_views,
            basename_index=rae_latent_basename_index,
            main_record=main_record,
            cache_record=cache_record,
            record_index=record_index,
            result_path=rae_latent_source_path,
        )
        mllm_correct = get_mllm_correct_from_main(
            main_record,
            source_path=main_path,
            record_index=record_index,
        )
        mllm_score_map = get_mllm_score_map_from_main(
            main_record,
            source_path=main_path,
            record_index=record_index,
        )

        fusion_pred, fusion_diag = fusion_predict(
            fsl_view=fsl_view,
            mllm_score_map=mllm_score_map,
            region=region,
            params=params,
        )

        fsl_correct = class_eq(fsl_view["top1"], ground_truth)
        fusion_correct = class_eq(fusion_pred, ground_truth)

        fsl_correct_count += int(fsl_correct)
        mllm_correct_count += int(mllm_correct)
        fusion_correct_count += int(fusion_correct)
        guard_count += int(
            bool(fusion_diag.get("used_guard", False))
        )
        override_count += int(
            bool(fusion_diag.get("used_override", False))
        )
        fsl_source_fields[fsl_view["source_field"]] += 1

        if not fsl_correct and fusion_correct:
            fixed_count += 1
        elif fsl_correct and not fusion_correct:
            broken_count += 1

        # 输出以非 cache JSON 的记录为主体。
        # MLLM 字段原样来自非 cache 文件。
        # FSL predictions are from Adaptive F1-v5 query_results.json.
        merged_record = dict(main_record)
        merged_record.update({
            "ground_truth": ground_truth,
            "fsl_mode": fsl_mode,
            "fsl_top1": fsl_view["top1"],
            "fsl_gap": fsl_view["gap"],
            "fsl_distance": fsl_view["distance"],
            "fsl_ranking": fsl_view["ranking"],
            "fsl_correct": fsl_correct,
            "fsl_source_json": str(rae_latent_source_path),
            "fsl_source_field": fsl_view["source_field"],
            "mllm_correct": mllm_correct,
            "mllm_source_json": str(main_path),
            # V2 fusion outputs
            "fusion_v2_pred": fusion_pred,
            "fusion_v2_correct": fusion_correct,
            "fusion_region": region,
            "fusion_strategy": params["strategy"],
            "fusion_params": dict(params),
            "fusion_guard_used": bool(
                fusion_diag.get("used_guard", False)
            ),
            "fusion_override_used": bool(
                fusion_diag.get("used_override", False)
            ),
            "fusion_mllm_prob_on_fsl_top1": (
                fusion_diag.get("mllm_prob_on_fsl_top1")
            ),
            "fusion_combined_scores": (
                fusion_diag.get("combined_scores", {})
            ),
            "fusion_diag": dict(fusion_diag),
        })
        output_records.append(merged_record)

    total = len(output_records)
    if total <= 0:
        raise ValueError(
            f"没有可输出记录：cache={cache_path}, main={main_path}"
        )

    summary = {
        **metadata,
        "fsl_mode": fsl_mode,
        "total": total,
        "fsl_correct": fsl_correct_count,
        "fsl_acc": fsl_correct_count / total,
        "mllm_correct": mllm_correct_count,
        "mllm_acc": mllm_correct_count / total,
        "fusion_correct": fusion_correct_count,
        "fusion_acc": fusion_correct_count / total,
        "fusion_minus_fsl_correct": (
            fusion_correct_count - fsl_correct_count
        ),
        "fusion_minus_mllm_correct": (
            fusion_correct_count - mllm_correct_count
        ),
        "fixed_fsl_errors": fixed_count,
        "broken_fsl_correct": broken_count,
        "guard_used": guard_count,
        "override_used": override_count,
        "alignment_method": alignment_method,
        "fsl_source_field_counts": dict(fsl_source_fields),

        # V2 organ-specific fusion
        "region": region,
        "strategy": params["strategy"],
        "param_alpha": params.get("alpha"),
        "param_guard": params.get("guard"),
        "param_top1_thr": params.get("top1_thr"),
        "param_fsl_thr": params.get("fsl_thr"),
        "param_gap_thr": params.get("gap_thr"),
        "fsl_cache_json": None,

        # NEW RAE latent-traj provenance
        "rae_fsl_source_json": (
            str(rae_latent_source_path)
            if rae_latent_source_path is not None
            else None
        ),
        "rae_fsl_source_kind": rae_latent_source_kind,
        "rae_fsl_mode": (
            RAE_LATENT_MODE
            if rae_latent_source_path is not None
            else None
        ),
        "rae_fsl_score": (
            RAE_LATENT_SCORE
            if rae_latent_source_path is not None
            else None
        ),
        "rae_fsl_support_agg": (
            RAE_LATENT_SUPPORT_AGG
            if rae_latent_source_path is not None
            else None
        ),

        "mllm_json": str(main_path),
        "output_json": str(output_path),
    }

    output_data = {
        "config": {
            "fsl_mode": fsl_mode,
            "region": region,
            "strategy": params["strategy"],
            "fusion_params": dict(params),
            "fsl_source_policy": "Adaptive F1-v5: branches.adaptive",
            "mllm_source_policy": "only paired non-cache JSON",
        },
        "sources": {
            "fsl_cache_json": None,
            "rae_fsl_source_json": (
                str(rae_latent_source_path)
                if rae_latent_source_path is not None
                else None
            ),
            "mllm_json": str(main_path),
            "cache_top_level_keys": sorted(cache_data.keys()),
            "mllm_top_level_keys": sorted(main_data.keys()),
        },
        "summary": summary,
        "results": output_records,
    }

    atomic_write_json(output_path, output_data)

    print(
        f"[DONE] {metadata['city']}/{metadata['organ']} "
        f"{metadata['space']} {fsl_mode}\n"
        f"       FSL source: "
        f"{rae_latent_source_path}\n"
        f"       MLLM JSON : {main_path}\n"
        f"       output    : {output_path}\n"
        f"       region={region} | strategy={params['strategy']} | "
        f"align={alignment_method} | "
        f"FSL={fsl_correct_count}/{total} "
        f"({fsl_correct_count / total:.4f}) | "
        f"MLLM={mllm_correct_count}/{total} "
        f"({mllm_correct_count / total:.4f}) | "
        f"Fusion={fusion_correct_count}/{total} "
        f"({fusion_correct_count / total:.4f}) | "
        f"delta_vs_FSL={fusion_correct_count-fsl_correct_count:+d}"
    )

    if verbose_details:
        print(
            f"       fixed={fixed_count}, broken={broken_count}, "
            f"guard={guard_count}, override={override_count}, "
            f"params={params}, "
            f"FSL fields={dict(fsl_source_fields)}"
        )

    return summary


# =============================================================================
# 文件配对与批处理
# =============================================================================
def collect_file_pairs(
    *,
    input_root: Path,
    output_root: Path,
) -> tuple[list[tuple[Path, Path]], list[dict[str, str]]]:
    """
    NO-CACHE mode.

    Directly traverse MLLM main JSON files under:
        <input_root>/<city>/<organ>/**/*.json

    The old return shape ``(cache_path, main_path)`` is intentionally kept for
    minimum downstream changes.  In this mode both entries point to the same
    MLLM main JSON, because image/latent FSL rankings are loaded from the
    separate RAW/RAE result roots rather than from *_fsl_cache.json.
    """
    pairs: list[tuple[Path, Path]] = []
    missing: list[dict[str, str]] = []

    ignored_names = {
        "batch_fusion_summary.json",
        "batch_fusion_failures.json",
        "missing_file_pairs.json",
    }

    for city in CITY_ORDER:
        city_root = input_root / city
        if not city_root.is_dir():
            print(f"[WARN] 城市目录不存在，跳过：{city_root}")
            continue

        for main_path in sorted(city_root.rglob("*.json")):
            if not main_path.is_file():
                continue
            if is_ignored_path(main_path, input_root):
                continue
            if path_is_inside(main_path, output_root):
                continue
            if main_path.name.endswith(CACHE_SUFFIX):
                continue
            if main_path.name in ignored_names:
                continue

            # Only process actual MLLM result JSONs.
            name_lower = main_path.name.lower()
            if "_image_" not in name_lower and "_latent_" not in name_lower:
                continue

            # Keep the historical tuple interface.  cache_path == main_path.
            pairs.append((main_path, main_path))

    return pairs, missing


def make_output_path(
    *,
    input_root: Path,
    output_root: Path,
    cache_path: Path,
    fsl_mode: str,
) -> Path:
    # NO-CACHE mode: cache_path is the MLLM main JSON itself.
    relative_parent = cache_path.parent.relative_to(input_root)
    base_stem = cache_path.stem
    return (
        output_root
        / relative_parent
        / f"{base_stem}_fusion_{fsl_mode}.json"
    )


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "city",
        "organ",
        "space",
        "fsl_mode",
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
        "guard_used",
        "override_used",
        "alignment_method",
        "region",
        "strategy",
        "param_alpha",
        "param_guard",
        "param_top1_thr",
        "param_fsl_thr",
        "param_gap_thr",
        "fsl_cache_json",
        "rae_fsl_source_json",
        "rae_fsl_source_kind",
        "rae_fsl_mode",
        "rae_fsl_score",
        "rae_fsl_support_agg",
        "mllm_json",
        "output_json",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "遍历苍溪/自贡/南充，使用 *_fsl_cache.json 配对同名 MLLM JSON；"
            "image/latent MLLM files both fuse with adaptive F1-v5 "
            "query_results.json results[*].branches.adaptive；"
            "融合采用 V2 器官特异策略。"
        )
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default=(
            "/mnt/inaisfs/data/home/tansy_criait/"
            "wass_flow_match_胃/mllm_fsl_result/result_other_0729"
        ),
        help=(
            "输入根目录。默认："
            "/mnt/inaisfs/data/home/tansy_criait/"
            "wass_flow_match_胃/mllm_fsl_result/result_other_0729"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "输出根目录。默认写入 "
            "<input_root>/fusion_traj_tlast_image_feat"
        ),
    )
    # -----------------------------------------------------------------
    # V2 organ-specific fusion parameters.
    # Each organ has independent CLI arguments.
    # -----------------------------------------------------------------
    parser.add_argument(
        "--stomach_alpha",
        type=float,
        required=True,
        help="胃 gap_guard_fusion: FSL 权重。",
    )
    parser.add_argument(
        "--stomach_guard",
        type=float,
        required=True,
        help="胃 gap_guard_fusion: MLLM(FSL-top1) guard。",
    )
    parser.add_argument(
        "--stomach_gap_thr",
        type=float,
        required=True,
        help="胃 gap_guard_fusion: FSL gap guard threshold。",
    )

    parser.add_argument(
        "--esophagus_top1_thr",
        type=float,
        required=True,
        help="食管 high_conf_override: MLLM top1 threshold。",
    )
    parser.add_argument(
        "--esophagus_fsl_thr",
        type=float,
        required=True,
        help="食管 high_conf_override: MLLM probability on FSL top1 threshold。",
    )
    parser.add_argument(
        "--esophagus_gap_thr",
        type=float,
        required=True,
        help="食管 high_conf_override: FSL gap threshold。",
    )

    parser.add_argument(
        "--duodenum_top1_thr",
        type=float,
        required=True,
        help="十二指肠 very_high_conf_override: MLLM top1 threshold。",
    )
    parser.add_argument(
        "--duodenum_fsl_thr",
        type=float,
        required=True,
        help="十二指肠 very_high_conf_override: MLLM probability on FSL top1 threshold。",
    )
    parser.add_argument(
        "--rae_fsl_result_root",
        default=None,
        help=(
            "Adaptive F1-v5 结果根目录，包含 <root>/<city>/<organ>/query_results.json。"
        ),
    )

    parser.add_argument(
        "--verbose_details",
        action="store_true",
        help="打印 fixed/broken/guard 等细节。",
    )
    parser.add_argument(
        "--fail_on_missing_pair",
        action="store_true",
        help="发现 cache 文件缺少对应非 cache JSON 时直接失败。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出；默认跳过已存在输出。",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"输入根目录不存在：{input_root}")

    # Validate organ-specific fusion parameters
    probability_args = {
        "--stomach_alpha": args.stomach_alpha,
        "--esophagus_top1_thr": args.esophagus_top1_thr,
        "--esophagus_fsl_thr": args.esophagus_fsl_thr,
        "--duodenum_top1_thr": args.duodenum_top1_thr,
        "--duodenum_fsl_thr": args.duodenum_fsl_thr,
    }
    for name, value in probability_args.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} 必须在 [0,1]：{value}"
            )

    if args.stomach_guard < 0.0:
        raise ValueError(
            f"--stomach_guard 必须非负：{args.stomach_guard}"
        )
    if args.stomach_gap_thr < 0.0:
        raise ValueError(
            f"--stomach_gap_thr 必须非负：{args.stomach_gap_thr}"
        )
    if args.esophagus_gap_thr < 0.0:
        raise ValueError(
            f"--esophagus_gap_thr 必须非负：{args.esophagus_gap_thr}"
        )

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else input_root / "fusion_traj_tlast_image_feat"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.rae_fsl_result_root:
        parser.error("--rae_fsl_result_root is required for Adaptive F1-v5 fusion")
    rae_fsl_result_root = Path(args.rae_fsl_result_root).expanduser().resolve()
    if (
        rae_fsl_result_root is not None
        and not rae_fsl_result_root.is_dir()
    ):
        raise NotADirectoryError(
            f"--rae_fsl_result_root 不存在：{rae_fsl_result_root}"
        )

    pairs, missing_pairs = collect_file_pairs(
        input_root=input_root,
        output_root=output_root,
    )

    if missing_pairs:
        print("\n[WARN] 以下 FSL cache 缺少对应的非 cache JSON：")
        for item in missing_pairs:
            print(
                f"  cache   : {item['fsl_cache_json']}\n"
                f"  expected: {item['expected_mllm_json']}"
            )

        atomic_write_json(
            output_root / "missing_file_pairs.json",
            missing_pairs,
        )

        if args.fail_on_missing_pair:
            raise FileNotFoundError(
                f"发现 {len(missing_pairs)} 个缺失文件对。"
            )

    if not pairs:
        raise RuntimeError(
            f"没有找到有效 MLLM 主 JSON：{input_root}"
        )

    print("=" * 100)
    print("[NO-CACHE BATCH FSL + MLLM]")
    print(f"input_root   : {input_root}")
    print(f"output_root  : {output_root}")
    print(f"cities       : {', '.join(CITY_ORDER)}")
    print(f"MLLM JSONs   : {len(pairs)}")
    print(f"modes        : {', '.join(FSL_MODES)}")
    print("[V2 ORGAN PARAMETERS]")
    print(
        "胃           : gap_guard_fusion | "
        f"alpha={args.stomach_alpha} | "
        f"guard={args.stomach_guard} | "
        f"gap_thr={args.stomach_gap_thr}"
    )
    print(
        "食管         : high_conf_override | "
        f"top1_thr={args.esophagus_top1_thr} | "
        f"fsl_thr={args.esophagus_fsl_thr} | "
        f"gap_thr={args.esophagus_gap_thr}"
    )
    print(
        "十二指肠     : very_high_conf_override | "
        f"top1_thr={args.duodenum_top1_thr} | "
        f"fsl_thr={args.duodenum_fsl_thr}"
    )
    print(f"RAE FSL root : {rae_fsl_result_root}")
    print(
        "RAE latent   : "
        f"{RAE_LATENT_MODE}/{RAE_LATENT_SUPPORT_AGG}/{RAE_LATENT_SCORE}"
    )
    print("=" * 100)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    skipped_outputs = 0

    total_jobs = len(pairs) * len(FSL_MODES)
    job_index = 0

    for cache_path, main_path in pairs:
        metadata = infer_metadata(
            input_root=input_root,
            cache_path=cache_path,
        )
        region = metadata["organ"]

        if region not in REGION_STRATEGIES:
            print(
                f"[ERROR] 无法识别支持的器官 region："
                f"organ={region!r}, file={cache_path}"
            )
            for fsl_mode in FSL_MODES:
                job_index += 1
                failures.append({
                    "fsl_cache_json": None,
                    "mllm_json": str(main_path),
                    "fsl_mode": fsl_mode,
                    "error": f"Unsupported organ/region: {region!r}",
                })
            continue

        params = build_region_params(
            region,
            args,
        )

        for fsl_mode in FSL_MODES:
            job_index += 1
            output_path = make_output_path(
                input_root=input_root,
                output_root=output_root,
                cache_path=cache_path,
                fsl_mode=fsl_mode,
            )

            print(
                f"\n[{job_index}/{total_jobs}] "
                f"{cache_path.relative_to(input_root)} "
                f"| organ={region} "
                f"| strategy={params['strategy']} "
                f"| mode={fsl_mode}"
            )

            if output_path.exists() and not args.overwrite:
                skipped_outputs += 1
                print(
                    f"[SKIP] 输出已存在：{output_path}\n"
                    "       需要覆盖时添加 --overwrite"
                )
                continue

            try:
                summary = process_pair_mode(
                    input_root=input_root,
                    cache_path=cache_path,
                    main_path=main_path,
                    output_path=output_path,
                    fsl_mode=fsl_mode,
                    region=region,
                    params=params,
                    verbose_details=args.verbose_details,
                    rae_fsl_result_root=rae_fsl_result_root,
                )
                summaries.append(summary)
            except Exception as exc:
                failure = {
                    "fsl_cache_json": None,
                    "mllm_json": str(main_path),
                    "fsl_mode": fsl_mode,
                    "error": repr(exc),
                }
                failures.append(failure)
                print(
                    f"[ERROR] mode={fsl_mode}, "
                    f"cache={cache_path}\n"
                    f"        {repr(exc)}"
                )

    summary_json_path = output_root / "batch_fusion_summary.json"
    summary_csv_path = output_root / "batch_fusion_summary.csv"
    failure_path = output_root / "batch_fusion_failures.json"

    batch_summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "cities": list(CITY_ORDER),
        "modes": list(FSL_MODES),
        "v2_organ_params": {
            "胃": build_region_params("胃", args),
            "食管": build_region_params("食管", args),
            "十二指肠": build_region_params("十二指肠", args),
        },
        "rae_fsl_result_root": (
            str(rae_fsl_result_root)
            if rae_fsl_result_root is not None
            else None
        ),
        "rae_latent_protocol": {
            "mode": RAE_LATENT_MODE,
            "score": RAE_LATENT_SCORE,
            "support_aggregation": RAE_LATENT_SUPPORT_AGG,
        },
        "discovered_mllm_jsons": len(pairs),
        "expected_jobs": total_jobs,
        "successful_jobs_this_run": len(summaries),
        "skipped_existing_outputs": skipped_outputs,
        "failed_jobs": len(failures),
        "missing_file_pairs": 0,
        "results": summaries,
        "failures": failures,
    }

    atomic_write_json(summary_json_path, batch_summary)
    write_summary_csv(summary_csv_path, summaries)
    atomic_write_json(failure_path, failures)

    print("\n" + "=" * 100)
    print("[BATCH DONE]")
    print(f"file pairs       : {len(pairs)}")
    print(f"expected jobs    : {total_jobs}")
    print(f"successful       : {len(summaries)}")
    print(f"skipped existing : {skipped_outputs}")
    print(f"failed           : {len(failures)}")
    print(f"missing pairs    : {len(missing_pairs)}")
    print(f"summary JSON     : {summary_json_path}")
    print(f"summary CSV      : {summary_csv_path}")
    print(f"failures JSON    : {failure_path}")
    print("=" * 100)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
