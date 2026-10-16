#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huaxi GastroFusion V2: MLLM main JSON + offline RAE Uniform trajectory FSL.

FSL is ALWAYS obtained from --rae_fsl_result_root:
  query_results.json: scores["uniform"]["min"]["tolast_mean"]
  result.json fallback: rae_metric["uniform"]["tolast"]["mean"], class min.
No cache, raw_fsl_result, fsl_result, fsl_full_result, Direct, Decay, or Kvasir.
All eight numerical fusion parameters MUST be supplied on the CLI.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

REGION_STRATEGIES = {
    "胃": "gap_guard_fusion",
    "食管": "high_conf_override",
    "十二指肠": "very_high_conf_override",
}
RAE_LATENT_MODE = "uniform"
RAE_LATENT_SCORE = "tolast_mean"
RAE_LATENT_SUPPORT_AGG = "min"
IGNORED_DIR_NAMES = {"备份", "backup", "backups", "bak", "__pycache__"}
IMAGE_PATH_KEYS = ("image_path", "query_path", "img_path", "path", "image")
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

    优先 query_results.json，找不到时 fallback 到 result.json。
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

    result_candidates = sorted(
        p.resolve()
        for p in organ_root.rglob("result.json")
        if p.is_file()
        and not any(
            str(part).lower() in IGNORED_DIR_NAMES
            for part in p.relative_to(organ_root).parts
        )
    )

    if not result_candidates:
        raise FileNotFoundError(
            "华西 NEW RAE organ 目录下既没有 query_results.json，"
            "也没有 result.json：\n"
            f"  organ={organ}\n"
            f"  search_dir={organ_root}"
        )

    if len(result_candidates) > 1:
        listing = "\n".join(
            f"    - {p}" for p in result_candidates
        )
        raise RuntimeError(
            "同一华西 organ 下找到多个 result.json，"
            "无法自动确定使用哪个：\n"
            f"  organ={organ}\n{listing}"
        )

    return result_candidates[0], "result"

def build_rae_latent_views_from_query_results(
    query_results_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    """
    直接读取：
        scores["uniform"]["min"]["tolast_mean"]
    """
    with query_results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("queries"), list)
    ):
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
            mode="RAE latent traj uniform/min/tolast_mean",
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

def _rae_uniform_tolast_mean_from_pair(
    record: dict[str, Any],
) -> float:
    evidence = record.get("rae_metric")
    if not isinstance(evidence, dict):
        raise KeyError("result.json row 缺少 rae_metric")

    try:
        value = (
            evidence[RAE_LATENT_MODE]
            ["tolast"]
            ["mean"]
        )
    except Exception as exc:
        raise KeyError(
            "result.json row 缺少 "
            "rae_metric['uniform']['tolast']['mean']"
        ) from exc

    value = safe_float(value)
    if value is None:
        raise ValueError(
            "rae_metric['uniform']['tolast']['mean'] "
            "不是有效数值"
        )

    return float(value)

def build_rae_latent_views_from_result_json(
    result_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    """
    fallback：从 pair-level result.json 重建
        uniform + tolast_mean + support-min
    """
    records = _load_rae_result_records(
        result_path
    )

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for record in records:
        key = _rae_raw_query_key(record)
        grouped.setdefault(key, []).append(record)

    views: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    basename_candidates: dict[
        str,
        list[tuple[str, str]],
    ] = {}

    for key, rows in grouped.items():
        class_values: dict[
            str,
            list[float],
        ] = {}
        gt = None

        for row in rows:
            value = _rae_uniform_tolast_mean_from_pair(
                row
            )
            class_name = _rae_candidate_class(row)

            class_values.setdefault(
                class_name,
                [],
            ).append(float(value))

            if gt is None:
                gt = record_ground_truth(row)

        ranking = [
            {
                "class": class_name,
                "distance": float(min(values)),
            }
            for class_name, values
            in class_values.items()
            if values
        ]
        ranking.sort(
            key=lambda item: item["distance"]
        )

        if not ranking:
            raise ValueError(
                "RAE result.json 无法构造 latent-traj ranking："
                f"file={result_path}, query={key}"
            )

        top1 = str(ranking[0]["class"])
        gap = calc_gap_from_ranking(ranking)

        views[key] = {
            "top1": top1,
            "gap": float(gap),
            "distance": float(
                ranking[0]["distance"]
            ),
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

        if key[0] == "path":
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

def build_rae_latent_views(
    *,
    source_path: Path,
    source_kind: str,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    if source_kind == "query_results":
        return build_rae_latent_views_from_query_results(
            source_path
        )

    if source_kind == "result":
        return build_rae_latent_views_from_result_json(
            source_path
        )

    raise ValueError(
        f"未知 RAE source_kind：{source_kind}"
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

def get_rae_fsl_view_for_record(
    *, rae_views: dict, basename_index: dict, main_record: dict,
    record_index: int, source_path: Path,
) -> dict[str, Any]:
    # Match by exact path, then explicit ID, then unique basename.
    for field in (*IMAGE_PATH_KEYS, "target_image", "x0_path"):
        value = main_record.get(field)
        if value:
            key = ("path", normalize_path_key(value))
            if key in rae_views:
                return rae_views[key]
    for field in ("data_id", "idx", "index", "query_idx", "question_id"):
        value = main_record.get(field)
        if value is not None:
            for raw_field in ("data_id", "idx", "index", "query_idx", "question_id"):
                key = (raw_field, str(value))
                if key in rae_views:
                    return rae_views[key]
    for field in (*IMAGE_PATH_KEYS, "target_image", "x0_path"):
        value = main_record.get(field)
        if value:
            basename = os.path.basename(normalize_path_key(value))
            key = basename_index.get(basename)
            if key is not None and key in rae_views:
                return rae_views[key]
    raise KeyError(f"Cannot align MLLM query to offline RAE: row={record_index}, source={source_path}")


def infer_metadata(input_root: Path, main_path: Path) -> dict[str, str]:
    relative = main_path.relative_to(input_root)
    parts = relative.parent.parts
    organ = next((part for part in reversed(parts) if part in REGION_STRATEGIES), None)
    if organ is None:
        organ = next((o for o in REGION_STRATEGIES if o in main_path.stem), None)
    if organ is None:
        raise ValueError(f"Cannot infer organ from MLLM path: {main_path}")
    city = parts[0] if len(parts) >= 2 and parts[0] in {"苍溪", "自贡", "南充", "华西"} else "华西"
    if city != "华西":
        raise ValueError("This script is Huaxi-only; external centres are unsupported")
    if "_image_" in main_path.name.lower():
        raise ValueError(f"Image-space JSON is not supported: {main_path}")
    if "_latent_" not in main_path.name.lower():
        raise ValueError(f"Expected latent MLLM JSON: {main_path}")
    return {"city": city, "organ": organ, "space": "latent", "region": organ}


def build_region_params(region: str, args: argparse.Namespace) -> dict[str, Any]:
    if region not in REGION_STRATEGIES:
        raise ValueError(f"Unsupported region: {region}")
    params = {"strategy": REGION_STRATEGIES[region]}
    if region == "胃":
        params.update(alpha=args.stomach_alpha, guard=args.stomach_guard,
                      gap_thr=args.stomach_gap_thr)
    elif region == "食管":
        params.update(top1_thr=args.esophagus_top1_thr,
                      fsl_thr=args.esophagus_fsl_thr,
                      gap_thr=args.esophagus_gap_thr)
    else:
        params.update(top1_thr=args.duodenum_top1_thr,
                      fsl_thr=args.duodenum_fsl_thr, gap_thr=None)
    return params


def collect_main_jsons(input_root: Path, output_root: Path) -> list[Path]:
    files = []
    for path in sorted(input_root.rglob("*.json")):
        if is_ignored_path(path, input_root) or path_is_inside(path, output_root):
            continue
        if path.name.endswith("_fsl_cache.json"):
            continue
        if "_latent_" not in path.name.lower() or "fusion" in path.name.lower():
            continue
        try:
            _, records = load_json_results(path)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if records and any(isinstance(r.get("mllm_score_map"), dict) for r in records):
            files.append(path)
    return files


def process_main_json(*, input_root: Path, main_path: Path, output_path: Path,
                      rae_fsl_result_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    main_data, records = load_json_results(main_path)
    metadata = infer_metadata(input_root, main_path)
    region = metadata["region"]
    params = build_region_params(region, args)
    rae_source_path, rae_source_kind = resolve_rae_latent_huaxi_source(
        result_root=rae_fsl_result_root, organ=region)
    rae_views, basename_index = build_rae_latent_views(
        source_path=rae_source_path, source_kind=rae_source_kind)
    output_records = []
    fsl_correct_count = mllm_correct_count = fusion_correct_count = 0
    fixed_count = broken_count = guard_count = override_count = 0
    fsl_source_fields = Counter()
    for i, main_record in enumerate(records):
        gt = record_ground_truth(main_record)
        if gt is None:
            raise KeyError(f"Missing ground_truth: {main_path}, row={i}")
        fsl_view = get_rae_fsl_view_for_record(
            rae_views=rae_views, basename_index=basename_index,
            main_record=main_record, record_index=i, source_path=rae_source_path)
        mllm_correct = get_mllm_correct_from_main(
            main_record, source_path=main_path, record_index=i)
        mllm_score_map = get_mllm_score_map_from_main(
            main_record, source_path=main_path, record_index=i)
        fusion_pred, diag = fusion_predict(
            fsl_view=fsl_view, mllm_score_map=mllm_score_map,
            region=region, params=params)
        fsl_correct = class_eq(fsl_view["top1"], gt)
        fusion_correct = class_eq(fusion_pred, gt)
        fsl_correct_count += int(fsl_correct)
        mllm_correct_count += int(mllm_correct)
        fusion_correct_count += int(fusion_correct)
        fixed_count += int(not fsl_correct and fusion_correct)
        broken_count += int(fsl_correct and not fusion_correct)
        guard_count += int(bool(diag.get("used_guard", False)))
        override_count += int(bool(diag.get("used_override", False)))
        fsl_source_fields[fsl_view["source_field"]] += 1
        # Keep all original MLLM fields except obsolete FSL fields.
        merged = {k: v for k, v in main_record.items()
                  if k not in {"raw_fsl_result", "fsl_result", "fsl_full_result",
                               "fsl_full_result_direct", "fsl_ranking", "fsl_top1"}}
        merged.update({
            "ground_truth": gt,
            "fusion_v2_region": region,
            "fusion_v2_strategy": params["strategy"],
            "fusion_v2_params": params,
            "fusion_v2_fsl_mode": "traj",
            "fusion_v2_fsl_pred": fsl_view["top1"],
            "fusion_v2_fsl_gap": fsl_view["gap"],
            "fusion_v2_fsl_distance": fsl_view["distance"],
            "fusion_v2_fsl_ranking": fsl_view["ranking"],
            "fusion_v2_fsl_correct": fsl_correct,
            "fusion_v2_mllm_correct": mllm_correct,
            "fusion_v2_pred": fusion_pred,
            "fusion_v2_correct": fusion_correct,
            "fusion_v2_diagnostics": diag,
            "fusion_v2_fsl_source_json": str(rae_source_path),
            "fusion_v2_fsl_source_field": fsl_view["source_field"],
            "fusion_v2_rae_source_kind": rae_source_kind,
            "fusion_v2_rae_mode": RAE_LATENT_MODE,
            "fusion_v2_rae_score": RAE_LATENT_SCORE,
            "fusion_v2_rae_support_agg": RAE_LATENT_SUPPORT_AGG,
            "fusion_v2_mllm_source_json": str(main_path),
        })
        output_records.append(merged)
    total = len(output_records)
    if total == 0:
        raise ValueError(f"Empty MLLM results: {main_path}")
    summary = {
        **metadata, "fsl_mode": "traj", "strategy": params["strategy"],
        "total": total,
        "fsl_correct": fsl_correct_count, "fsl_acc": fsl_correct_count / total,
        "mllm_correct": mllm_correct_count, "mllm_acc": mllm_correct_count / total,
        "fusion_correct": fusion_correct_count, "fusion_acc": fusion_correct_count / total,
        "fusion_minus_fsl_correct": fusion_correct_count - fsl_correct_count,
        "fusion_minus_mllm_correct": fusion_correct_count - mllm_correct_count,
        "fixed_fsl_errors": fixed_count, "broken_fsl_correct": broken_count,
        "net_fixed": fixed_count - broken_count,
        "guard_used": guard_count, "override_used": override_count,
        "alignment_method": "mllm_main_direct_to_rae",
        "fsl_source_field_counts": dict(fsl_source_fields),
        "alpha": params.get("alpha"), "guard": params.get("guard"),
        "top1_thr": params.get("top1_thr"), "fsl_thr": params.get("fsl_thr"),
        "gap_thr": params.get("gap_thr"),
        "rae_fsl_source_json": str(rae_source_path),
        "rae_fsl_source_kind": rae_source_kind,
        "rae_fsl_mode": RAE_LATENT_MODE,
        "rae_fsl_score": RAE_LATENT_SCORE,
        "rae_fsl_support_agg": RAE_LATENT_SUPPORT_AGG,
        "mllm_json": str(main_path), "output_json": str(output_path),
    }
    atomic_write_json(output_path, {
        "config": {"region": region, "fsl_mode": "traj", "strategy": params["strategy"],
                   "params": params, "fsl_source_policy": "NEW RAE uniform/min/tolast_mean",
                   "mllm_source_policy": "main-only MLLM JSON"},
        "sources": {"rae_fsl_source_json": str(rae_source_path),
                    "mllm_json": str(main_path),
                    "main_top_level_keys": sorted(str(k) for k in main_data.keys()),
                    "alignment_method": "mllm_main_direct_to_rae"},
        "summary": summary, "results": output_records,
    })
    print(f"[DONE] {region}: FSL={fsl_correct_count}/{total}, "
          f"MLLM={mllm_correct_count}/{total}, Fusion={fusion_correct_count}/{total}")
    if args.verbose_details:
        print(f"       fixed={fixed_count}, broken={broken_count}, "
              f"guard={guard_count}, override={override_count}")
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["city", "organ", "space", "region", "fsl_mode", "strategy", "total",
              "fsl_correct", "fsl_acc", "mllm_correct", "mllm_acc", "fusion_correct",
              "fusion_acc", "fusion_minus_fsl_correct", "fusion_minus_mllm_correct",
              "fixed_fsl_errors", "broken_fsl_correct", "net_fixed", "guard_used",
              "override_used", "alignment_method", "alpha", "guard", "top1_thr",
              "fsl_thr", "gap_thr", "rae_fsl_source_json", "rae_fsl_source_kind",
              "rae_fsl_mode", "rae_fsl_score", "rae_fsl_support_agg",
              "mllm_json", "output_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", help="Directory containing MLLM main JSON files")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--rae_fsl_result_root", required=True)
    # Retain shell compatibility but forbid anything except Uniform latent-traj.
    parser.add_argument("--fsl_mode", choices=["traj"], default="traj")
    parser.add_argument("--space", choices=["latent"], default="latent")
    for name in ("stomach_alpha", "stomach_guard", "stomach_gap_thr",
                 "esophagus_top1_thr", "esophagus_fsl_thr", "esophagus_gap_thr",
                 "duodenum_top1_thr", "duodenum_fsl_thr"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--verbose_details", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("stomach_alpha", "esophagus_top1_thr", "esophagus_fsl_thr",
                 "duodenum_top1_thr", "duodenum_fsl_thr"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name} must be finite and in [0,1]")
    for name in ("stomach_guard", "stomach_gap_thr", "esophagus_gap_thr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rae_root = Path(args.rae_fsl_result_root).expanduser().resolve()
    for directory in (input_root, rae_root):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    output_root.mkdir(parents=True, exist_ok=True)
    main_files = collect_main_jsons(input_root, output_root)
    if not main_files:
        raise RuntimeError(f"No valid latent MLLM main JSON files found in {input_root}")
    summaries, failures = [], []
    skipped = 0
    for i, main_path in enumerate(main_files, 1):
        out_path = output_root / main_path.parent.relative_to(input_root) / (
            main_path.stem + "_fusion_v2_traj.json")
        print(f"[{i}/{len(main_files)}] {main_path.relative_to(input_root)}")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[SKIP] {out_path}")
            continue
        try:
            summaries.append(process_main_json(
                input_root=input_root, main_path=main_path, output_path=out_path,
                rae_fsl_result_root=rae_root, args=args))
        except Exception as exc:
            print(f"[ERROR] {main_path}: {exc!r}")
            failures.append({"mllm_json": str(main_path), "error": repr(exc)})
    batch = {
        "input_root": str(input_root), "output_root": str(output_root),
        "rae_fsl_result_root": str(rae_root),
        "rae_latent_protocol": {"mode": RAE_LATENT_MODE, "score": RAE_LATENT_SCORE,
                                "support_aggregation": RAE_LATENT_SUPPORT_AGG},
        "v2_organ_params": {o: build_region_params(o, args) for o in REGION_STRATEGIES},
        "discovered_main_jsons": len(main_files), "expected_jobs": len(main_files),
        "successful_jobs_this_run": len(summaries),
        "skipped_existing_outputs": skipped, "failed_jobs": len(failures),
        "results": summaries, "failures": failures,
    }
    atomic_write_json(output_root / "batch_fusion_v2_summary.json", batch)
    write_summary_csv(output_root / "batch_fusion_v2_summary.csv", summaries)
    atomic_write_json(output_root / "batch_fusion_v2_failures.json", failures)
    print(f"[BATCH DONE] success={len(summaries)}, skipped={skipped}, failed={len(failures)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
