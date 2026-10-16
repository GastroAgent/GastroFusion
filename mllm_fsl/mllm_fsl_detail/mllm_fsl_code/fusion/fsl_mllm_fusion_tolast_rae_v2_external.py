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

器官参数可从命令行分别指定；不同器官互不覆盖。


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


# New RAE latent-traj FSL protocol.  Keep these fixed so that the fusion FSL
# baseline exactly matches score_summary.csv:
#   mode=uniform, score=tolast_mean, aggregation=min
RAE_LATENT_MODE = "uniform"
RAE_LATENT_SCORE = "tolast_mean"
RAE_LATENT_SUPPORT_AGG = "min"


def load_raw_result_records(path: Path) -> list[dict[str, Any]]:
    """
    读取上游 pairwise result.json。

    支持：
      1) 顶层直接为 list
      2) 顶层为 {"results": [...]}

    每一条通常对应：
      query(target_image) × support/matched_class
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        records = data["results"]
    else:
        raise ValueError(
            "raw FSL result.json 顶层必须是 list 或包含 results 列表的 dict："
            f"{path}"
        )

    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(records):
        if not isinstance(item, dict):
            raise TypeError(
                f"raw FSL result[{i}] 不是 dict：{path}"
            )
        cleaned.append(item)
    return cleaned


def raw_query_key(record: dict[str, Any]) -> tuple[str, str]:
    """
    为 raw result.json 的 pairwise 行生成 query key。

    重要：优先使用 query-level ID，而不是绝对路径。
    当前项目中 MLLM results[*].idx 与离线 FSL 的 data_id/question_id
    对应；不同流水线的图像根目录可能不同，因此 ID 是更稳定的主键。

    若没有 ID，再回退到 target_image/x0_path/x0 等路径。
    """
    for field in ("data_id", "question_id", "query_idx", "idx", "index"):
        value = record.get(field)
        if value is not None:
            return (field, str(value))

    for field in ("target_image", "x0_path", "x0", "query_path", "image_path"):
        value = record.get(field)
        if value:
            return ("path", normalize_path_key(value))

    raise KeyError(
        "raw FSL result 记录无法确定 query："
        "缺少 data_id/question_id/idx 或 target_image/x0_path/x0 等字段。"
    )


def raw_candidate_class(record: dict[str, Any]) -> str:
    """
    raw result 中 support 所属候选类别。

    优先 matched_class；兼容 label_B/candidate_class/support_class。
    """
    for field in ("matched_class", "label_B", "candidate_class", "support_class"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)

    raise KeyError(
        "raw FSL result 记录缺少候选类别字段 "
        "(matched_class / label_B / candidate_class / support_class)"
    )






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




# =============================================================================
# New RAE offline result -> latent-traj FSL ranking
# Fixed protocol: uniform + tolast_mean + min
# =============================================================================
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

    Priority:
        1) query_results.json
        2) result.json

    We prefer query_results.json because it already contains the exact
    per-query class ranking for:
        scores["uniform"]["min"]["tolast_mean"]

    If query_results.json is absent, result.json is used to reconstruct the same
    view from pair-level rae_metric evidence with class-level support-min.
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


def build_rae_latent_views_from_query_results(
    query_results_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    """
    Read the exact RAE latent-traj view from query_results.json:

        scores["uniform"]["min"]["tolast_mean"]

    Besides the canonical path key, build root-independent fallback aliases so
    MLLM JSON under data/数据_merge can align to RAE results produced from
    cross_domain_balanced/... without requiring identical absolute roots.
    """
    with query_results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
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
            mode="RAE latent traj uniform/min/tolast_mean",
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


def _rae_uniform_tolast_mean_from_pair(record: dict[str, Any]) -> float:
    """
    Extract pair-level RAE score equivalent to:
        mode=uniform, score=tolast_mean
    from the enriched result.json row.
    """
    evidence = record.get("rae_metric")
    if not isinstance(evidence, dict):
        raise KeyError("result.json row 缺少 rae_metric")

    try:
        value = evidence[RAE_LATENT_MODE]["tolast"]["mean"]
    except Exception as exc:
        raise KeyError(
            "result.json row 缺少 rae_metric['uniform']['tolast']['mean']"
        ) from exc

    out = safe_float(value)
    if out is None:
        raise ValueError(
            "rae_metric['uniform']['tolast']['mean'] 不是有效数值"
        )
    return float(out)


def build_rae_latent_views_from_result_json(
    result_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    """
    Fallback for RAE runs without query_results.json.

    Reconstruct exactly uniform + tolast_mean + support-min, using the same robust
    query alignment index as the image-space raw FSL branch.
    """
    records = load_raw_result_records(result_path)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = raw_query_key(record)
        grouped.setdefault(key, []).append(record)

    views: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_candidates: dict[str, set[tuple[str, str]]] = {}

    for key, rows in grouped.items():
        class_values: dict[str, list[float]] = {}
        gt = None

        for row_index, row in enumerate(rows):
            try:
                value = _rae_uniform_tolast_mean_from_pair(row)
            except Exception as exc:
                raise type(exc)(
                    f"{exc} | file={result_path}, query_key={key}, row={row_index}"
                ) from exc

            class_name = raw_candidate_class(row)
            class_values.setdefault(class_name, []).append(float(value))

            if gt is None:
                gt = record_ground_truth(row)

        ranking = [
            {
                "class": class_name,
                "distance": float(min(values)),
            }
            for class_name, values in class_values.items()
            if values
        ]
        ranking.sort(key=lambda x: x["distance"])

        if not ranking:
            raise ValueError(
                "RAE result.json 无法构造 latent-traj ranking："
                f"file={result_path}, query_key={key}"
            )

        top1 = str(ranking[0]["class"])
        gap = calc_gap_from_ranking(ranking)

        views[key] = {
            "top1": top1,
            "gap": float(gap),
            "distance": float(ranking[0]["distance"]),
            "ranking": ranking,
            "source_field": (
                "rae_result:rae_metric/"
                f"{RAE_LATENT_MODE}/tolast/mean"
                f"+support_{RAE_LATENT_SUPPORT_AGG}"
            ),
            "mode": "traj",
            "support_agg": RAE_LATENT_SUPPORT_AGG,
            "ground_truth": gt,
        }

        for row in rows:
            row_gt = record_ground_truth(row) or gt
            for path_value in _query_path_values(row):
                _add_unique_fallback_candidate(
                    fallback_candidates,
                    _class_basename_alias(row_gt, path_value),
                    key,
                )
                _add_unique_fallback_candidate(
                    fallback_candidates,
                    _path_tail_alias(path_value, 4),
                    key,
                )
                _add_unique_fallback_candidate(
                    fallback_candidates,
                    f"basename::{os.path.basename(path_value).casefold()}",
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

    if source_kind == "result":
        return build_rae_latent_views_from_result_json(source_path)

    raise ValueError(f"未知 RAE latent source_kind：{source_kind}")


# =============================================================================
# FSL 读取：严格从 cache 记录读取
# =============================================================================


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




def calc_gap_from_ranking(ranking: list[dict[str, Any]]) -> float:
    if len(ranking) > 1:
        return float(ranking[1]["distance"] - ranking[0]["distance"])
    return 999.0




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
    *, input_root: Path, main_path: Path, output_path: Path,
    region: str, params: dict[str, Any], verbose_details: bool,
    rae_fsl_result_root: Path,
) -> dict[str, Any]:
    main_data, main_records = load_json_results(main_path)
    metadata = infer_metadata(input_root=input_root, cache_path=main_path)
    if metadata["space"] != "latent" or region != metadata["organ"]:
        raise ValueError(f"Expected latent main JSON for {region}: {main_path}")
    rae_latent_source_path, rae_latent_source_kind = resolve_rae_latent_result_source(
        result_root=rae_fsl_result_root,
        city=metadata["city"], organ=metadata["organ"],
    )
    rae_latent_views, rae_latent_basename_index = build_rae_latent_views(
        source_path=rae_latent_source_path, source_kind=rae_latent_source_kind,
    )
    output_records = []
    fsl_correct_count = mllm_correct_count = fusion_correct_count = 0
    fixed_count = broken_count = guard_count = override_count = 0
    fsl_source_fields: Counter[str] = Counter()
    for record_index, main_record in enumerate(main_records):
        ground_truth = record_ground_truth(main_record)
        if ground_truth is None:
            raise KeyError(f"Missing ground_truth in {main_path}, row={record_index}")
        fsl_view = get_raw_fsl_view_for_record(
            raw_views=rae_latent_views,
            basename_index=rae_latent_basename_index,
            main_record=main_record,
            cache_record=main_record,
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
        # image-traj FSL 来自 raw Flow result；latent-traj FSL 来自 NEW RAE。
        merged_record = {k: v for k, v in main_record.items() if k not in ("raw_fsl_result", "fsl_result", "fsl_full_result")}
        merged_record.update({
            "ground_truth": ground_truth,
            "fsl_mode": "traj",
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
    if total == 0:
        raise ValueError(f"No results in {main_path}")
    summary = {
        **metadata, "fsl_mode": "traj", "total": total,
        "fsl_correct": fsl_correct_count, "fsl_acc": fsl_correct_count / total,
        "mllm_correct": mllm_correct_count, "mllm_acc": mllm_correct_count / total,
        "fusion_correct": fusion_correct_count, "fusion_acc": fusion_correct_count / total,
        "fusion_minus_fsl_correct": fusion_correct_count - fsl_correct_count,
        "fusion_minus_mllm_correct": fusion_correct_count - mllm_correct_count,
        "fixed_fsl_errors": fixed_count, "broken_fsl_correct": broken_count,
        "guard_used": guard_count, "override_used": override_count,
        "alignment_method": "mllm_main_to_offline_rae",
        "fsl_source_field_counts": dict(fsl_source_fields),
        "region": region, "strategy": params["strategy"],
        "param_alpha": params.get("alpha"), "param_guard": params.get("guard"),
        "param_top1_thr": params.get("top1_thr"), "param_fsl_thr": params.get("fsl_thr"),
        "param_gap_thr": params.get("gap_thr"),
        "rae_fsl_source_json": str(rae_latent_source_path),
        "rae_fsl_source_kind": rae_latent_source_kind,
        "rae_fsl_mode": RAE_LATENT_MODE, "rae_fsl_score": RAE_LATENT_SCORE,
        "rae_fsl_support_agg": RAE_LATENT_SUPPORT_AGG,
        "mllm_json": str(main_path), "output_json": str(output_path),
    }
    output_data = {
        "config": {
            "fsl_mode": "traj", "region": region, "strategy": params["strategy"],
            "fusion_params": dict(params),
            "fsl_source_policy": "latent-traj NEW RAE:uniform/min/tolast_mean",
            "mllm_source_policy": "main JSON only",
        },
        "sources": {
            "rae_fsl_source_json": str(rae_latent_source_path),
            "mllm_json": str(main_path),
            "mllm_top_level_keys": sorted(main_data.keys()),
        },
        "summary": summary, "results": output_records,
    }
    atomic_write_json(output_path, output_data)
    print(f"[DONE] {metadata['city']}/{region}: FSL={fsl_correct_count}/{total}, MLLM={mllm_correct_count}/{total}, Fusion={fusion_correct_count}/{total} -> {output_path}")
    if verbose_details:
        print(f"fixed={fixed_count}, broken={broken_count}, guard={guard_count}, override={override_count}")
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
            if "_latent_" not in name_lower:
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
    parser = argparse.ArgumentParser(description="External-centre Uniform RAE latent-trajectory + GastroMLLM V2 fusion")
    parser.add_argument("input_root", help="MLLM main JSON root, organized as city/organ")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--rae_fsl_result_root", required=True)
    for name in (
        "stomach_alpha", "stomach_guard", "stomach_gap_thr",
        "esophagus_top1_thr", "esophagus_fsl_thr", "esophagus_gap_thr",
        "duodenum_top1_thr", "duodenum_fsl_thr",
    ):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--verbose_details", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rae_root = Path(args.rae_fsl_result_root).expanduser().resolve()
    for p in (input_root, rae_root):
        if not p.is_dir():
            parser.error(f"Input directory does not exist: {p}")
    for name in ("stomach_alpha", "esophagus_top1_thr", "esophagus_fsl_thr", "duodenum_top1_thr", "duodenum_fsl_thr"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name} must be in [0,1]")
    for name in ("stomach_guard", "stomach_gap_thr", "esophagus_gap_thr"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    output_root.mkdir(parents=True, exist_ok=True)
    pairs, _ = collect_file_pairs(input_root=input_root, output_root=output_root)
    if not pairs:
        raise RuntimeError(f"No latent MLLM main JSON found in {input_root}")
    summaries = []
    failures = []
    skipped = 0
    for _, main_path in pairs:
        metadata = infer_metadata(input_root=input_root, cache_path=main_path)
        region = metadata["organ"]
        if region not in REGION_STRATEGIES:
            failures.append({"mllm_json": str(main_path), "error": f"Unsupported organ: {region}"})
            continue
        output_path = make_output_path(
            input_root=input_root, output_root=output_root,
            cache_path=main_path, fsl_mode="traj",
        )
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            summaries.append(process_pair_mode(
                input_root=input_root, main_path=main_path,
                output_path=output_path, region=region,
                params=build_region_params(region, args),
                verbose_details=args.verbose_details,
                rae_fsl_result_root=rae_root,
            ))
        except Exception as exc:
            failures.append({"mllm_json": str(main_path), "error": repr(exc)})
            print(f"[ERROR] {main_path}: {exc}")
    summary = {
        "input_root": str(input_root), "output_root": str(output_root),
        "rae_fsl_result_root": str(rae_root),
        "rae_latent_protocol": {"mode": "uniform", "score": "tolast_mean", "support_aggregation": "min"},
        "v2_organ_params": {region: build_region_params(region, args) for region in REGION_STRATEGIES},
        "discovered_mllm_jsons": len(pairs), "expected_jobs": len(pairs),
        "successful_jobs_this_run": len(summaries), "skipped_existing_outputs": skipped,
        "failed_jobs": len(failures), "results": summaries, "failures": failures,
    }
    atomic_write_json(output_root / "batch_fusion_summary.json", summary)
    write_summary_csv(output_root / "batch_fusion_summary.csv", summaries)
    atomic_write_json(output_root / "batch_fusion_failures.json", failures)
    print(f"[BATCH DONE] successful={len(summaries)}, skipped={skipped}, failed={len(failures)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
