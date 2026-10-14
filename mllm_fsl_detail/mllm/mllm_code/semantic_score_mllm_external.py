#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for GastroMLLM Candidate Service.

目标
----
1. 从 agent JSON / JSON glob 读取 query；
2. 调用：
      POST /v1/vision/candidate_diseases
3. 保存：
      - candidate_diseases
      - FULL-SET mllm_score_map
      - mllm_diag
      - mllm_correct
4. 输出 JSON 可直接作为当前 V2 fusion 的“非-cache main JSON”使用。

重要
----
- 本脚本不执行 FSL。
- candidate_diseases 不用于裁剪 FSL。
- FSL 仍由 fusion 脚本从 offline full-set RAE 结果读取。
- 本脚本只负责生成 MLLM 分支结果。
"""

from __future__ import annotations

import glob
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =============================================================================
# 1. 配置：通常只需要改这里
# =============================================================================

# MLLM candidate service
MLLM_API_BASE = "http://gpu35:8030"
MLLM_ENDPOINT = MLLM_API_BASE.rstrip("/") + "/v1/vision/candidate_diseases"

# -------------------------------------------------------------------------
# 输入
#
# 支持：
#   1) 单个 JSON
#   2) glob
#
# 示例：
# INPUT_GLOB = "/path/to/胃_agent.json"
# INPUT_GLOB = "/path/to/*_agent.json"
# INPUT_GLOB = "/path/to/**/*.json"
# -------------------------------------------------------------------------
INPUT_GLOB = (
    "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy_deepemd/huaxi_mllm_fsl_agent/external9_agent831_lot_way/*_agent.json"
)

# -------------------------------------------------------------------------
# 输出根目录
#
# 目标布局（external 9 datasets）：
# OUTPUT_ROOT/
#   自贡/
#     胃/
#       Med-qwen3-vl-RL_latent_自贡_胃.json
#     食管/
#       Med-qwen3-vl-RL_latent_自贡_食管.json
#     十二指肠/
#       Med-qwen3-vl-RL_latent_自贡_十二指肠.json
#   苍溪/
#     ...
#   南充/
#     ...
#
# 这样 3 城市 × 3 器官 = 9 个 JSON，不会互相覆盖。
# -------------------------------------------------------------------------
OUTPUT_ROOT = Path(
    "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/mllm_result_external9_lot_way1"
)

OUTPUT_FILENAME_TEMPLATE = "Med-qwen3-vl-RL_latent_{city}_{organ}.json"

# MLLM 输出 Top-k，仅作为诊断信息；不裁剪 FSL
TOP_K = 5

# 必须 True：
# 当前 V2 fusion 依赖 FULL-SET mllm_score_map[class]["norm_prob"]
USE_TEACHER_FORCING = True

SCORE_SOFTMAX_TEMPERATURE = 1.0
LENGTH_NORMALIZE = True

# 单条请求
REQUEST_TIMEOUT = 600
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3.0

# 每处理多少条落盘一次
SAVE_EVERY = 1

# True：如果已有输出，自动读取已经成功的 query，然后断点续跑
RESUME = True

# True：即使已有成功结果也重跑
OVERWRITE_EXISTING_QUERY = False

# 如果请求失败，是否保留 failure record 后继续
CONTINUE_ON_ERROR = True


# =============================================================================
# 2. 基础 IO
# =============================================================================

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def extract_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = (
            data.get("results")
            or data.get("data")
            or data.get("records")
            or data.get("items")
            or []
        )
    else:
        raise TypeError(
            f"JSON 顶层必须是 list 或 dict，当前为 {type(data).__name__}"
        )

    if not isinstance(records, list):
        raise TypeError("无法从 JSON 中解析 records/results/data 列表")

    out = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise TypeError(f"records[{i}] 不是 dict")
        out.append(r)
    return out


# =============================================================================
# 3. Record 解析
# =============================================================================

def first_nonempty(record: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        v = record.get(k)
        if v is not None and v != "":
            return v
    return None


def get_image_path(record: Dict[str, Any]) -> Optional[str]:
    value = first_nonempty(
        record,
        (
            "target_image",
            "image_path",
            "query_path",
            "x0",
            "img",
            "image",
            "path",
        ),
    )
    return str(value).strip() if value is not None else None


def get_ground_truth(record: Dict[str, Any]) -> Optional[str]:
    value = first_nonempty(
        record,
        (
            "ground_truth_name",
            "ground_truth",
            "gt",
            "label",
            "true_label",
            "long_label",
        ),
    )
    return str(value).strip() if value is not None else None


def normalize_class(value: Any) -> str:
    """
    与当前 fusion 脚本 class_eq 的基本语义保持一致：
    去空格 + lower。
    """
    if value is None:
        return ""
    return str(value).replace(" ", "").strip().lower()


def class_eq(a: Any, b: Any) -> bool:
    return normalize_class(a) == normalize_class(b)


def infer_organ_from_record_or_path(
    record: Dict[str, Any],
    source_json: str | Path,
) -> str:
    # record 优先
    for key in ("organ", "part", "region", "site"):
        value = record.get(key)
        if value:
            text = str(value)
            if "十二指肠" in text:
                return "十二指肠"
            if "食管" in text:
                return "食管"
            if "胃" in text:
                return "胃"

    # 再从 JSON 文件名 / path 推断
    text = str(source_json)
    if "十二指肠" in text:
        return "十二指肠"
    if "食管" in text:
        return "食管"
    if "胃" in text:
        return "胃"

    raise ValueError(
        f"无法推断 organ。请在 record 中增加 organ，或让文件路径包含胃/食管/十二指肠：{source_json}"
    )


def infer_city_from_record_or_path(
    record: Dict[str, Any],
    source_json: str | Path,
) -> str:
    """
    外院 9 组按 city + organ 唯一保存。

    优先从 record 中读取 city/hospital/center 等字段，
    否则从源 JSON 路径或文件名中识别：
      自贡 / 苍溪 / 南充 / 华西
    """
    city_names = ("自贡", "苍溪", "南充", "华西")

    for key in (
        "city",
        "hospital",
        "hospital_name",
        "center",
        "centre",
        "source_city",
        "source_hospital",
    ):
        value = record.get(key)
        if value:
            value_text = str(value)
            for city in city_names:
                if city in value_text:
                    return city

    path_text = str(source_json)
    for city in city_names:
        if city in path_text:
            return city

    raise ValueError(
        "无法推断 city。请确保 record 中含 city/hospital/center，"
        "或者输入 JSON 文件名/路径中包含 自贡/苍溪/南充/华西："
        f"{source_json}"
    )


def normalize_support_classes(
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    将 agent JSON 的 support_classes 转成 service 需要的格式：
      {
        "label_name": ...,
        "support_dir": ...,
        "visual_profile": ...,
        "aliases": [...]
      }
    """
    raw = record.get("support_classes") or []

    if not isinstance(raw, list):
        raise TypeError("support_classes 必须是 list")

    out: List[Dict[str, Any]] = []
    seen = set()

    for item in raw:
        if not isinstance(item, dict):
            continue

        label = (
            item.get("label_name")
            or item.get("label")
            or item.get("class_name")
            or item.get("name")
        )
        if not label:
            continue

        label = str(label).strip()
        if not label or label in seen:
            continue
        seen.add(label)

        support_dir = (
            item.get("support_dir")
            or item.get("dir")
            or item.get("path")
        )

        visual_profile = (
            item.get("visual_profile")
            or item.get("profile")
            or item.get("description")
        )

        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []

        out.append(
            {
                "label_name": label,
                "support_dir": (
                    str(support_dir).strip()
                    if support_dir is not None
                    else None
                ),
                "visual_profile": (
                    str(visual_profile).strip()
                    if visual_profile
                    else None
                ),
                "aliases": [str(x).strip() for x in aliases if str(x).strip()],
            }
        )

    if not out:
        raise ValueError("当前 record 没有有效 support_classes")

    return out


# =============================================================================
# 4. Service client
# =============================================================================

class MLLMCandidateClient:
    def __init__(
        self,
        endpoint: str,
        timeout: int = 600,
        max_retries: int = 3,
        retry_sleep: float = 3.0,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()

    def healthcheck(self) -> Dict[str, Any]:
        base = self.endpoint.split("/v1/vision/candidate_diseases")[0]
        resp = self.session.get(
            base.rstrip("/") + "/",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def infer(
        self,
        *,
        target_image: str,
        support_classes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        payload = {
            "target_image": target_image,
            "support_classes": support_classes,
            "top_k": TOP_K,
            "use_teacher_forcing": USE_TEACHER_FORCING,
            "softmax_temperature": SCORE_SOFTMAX_TEMPERATURE,
            "length_normalize": LENGTH_NORMALIZE,
        }

        last_exc = None

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.time()
                resp = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                elapsed = time.time() - t0

                if not isinstance(data, dict):
                    raise TypeError(
                        f"Service 返回不是 dict: {type(data).__name__}"
                    )

                if not data.get("ok", False):
                    raise RuntimeError(
                        f"Service 返回 ok=False: {data}"
                    )

                score_map = data.get("mllm_score_map")
                if not isinstance(score_map, dict) or not score_map:
                    raise KeyError(
                        "Service 返回中缺少有效 FULL-SET mllm_score_map"
                    )

                data["_time_cost"] = elapsed
                return data

            except Exception as exc:
                last_exc = exc
                print(
                    f"    [WARN] request attempt "
                    f"{attempt}/{self.max_retries} failed: {repr(exc)}"
                )

                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)

        raise RuntimeError(
            f"MLLM service request failed after {self.max_retries} attempts"
        ) from last_exc


# =============================================================================
# 5. Resume / output helpers
# =============================================================================

def record_query_key(record: Dict[str, Any]) -> str:
    p = get_image_path(record)
    if p:
        return os.path.normpath(p).replace("\\", "/")

    for key in ("idx", "index", "query_idx", "data_id", "question_id", "id"):
        if record.get(key) is not None:
            return f"{key}:{record[key]}"

    return ""


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.is_file() or not RESUME:
        return []

    try:
        data = load_json(output_path)
        return extract_records(data)
    except Exception as exc:
        print(
            f"[WARN] 无法读取已有输出，忽略 resume："
            f"{output_path} | {repr(exc)}"
        )
        return []


def build_output_document(
    *,
    source_json: Path,
    city: str,
    organ: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    success = sum(
        1 for r in results
        if r.get("mllm_service_success") is True
    )
    failure = len(results) - success

    correct_values = [
        r.get("mllm_correct")
        for r in results
        if isinstance(r.get("mllm_correct"), bool)
    ]
    correct = sum(int(x) for x in correct_values)

    return {
        "stage": "mllm_candidate_service_full_scoremap",
        "city": city,
        "organ": organ,
        "source_json": str(source_json),
        "mllm_endpoint": MLLM_ENDPOINT,
        "top_k": TOP_K,
        "use_teacher_forcing": USE_TEACHER_FORCING,
        "score_softmax_temperature": SCORE_SOFTMAX_TEMPERATURE,
        "length_normalize": LENGTH_NORMALIZE,
        "summary": {
            "total": len(results),
            "success": success,
            "failure": failure,
            "mllm_correct": correct,
            "mllm_acc": (
                correct / len(correct_values)
                if correct_values
                else None
            ),
        },
        "results": results,
    }


# =============================================================================
# 6. 单个 agent JSON
# =============================================================================

def process_json(
    *,
    source_json: Path,
    client: MLLMCandidateClient,
) -> Path:

    raw_data = load_json(source_json)
    records = extract_records(raw_data)

    if not records:
        raise ValueError(f"输入 JSON 没有 records: {source_json}")

    # 一个 agent JSON 对应一个 city-organ dataset。
    city = infer_city_from_record_or_path(
        records[0],
        source_json,
    )
    organ = infer_organ_from_record_or_path(
        records[0],
        source_json,
    )

    # IMPORTANT:
    # 不能只按 organ 保存，否则 自贡/苍溪/南充 的同一器官会相互覆盖。
    #
    # 外院 9 组保存为：
    #   OUTPUT_ROOT/
    #     自贡/胃/Med-qwen3-vl-RL_latent_自贡_胃.json
    #     自贡/食管/...
    #     ...
    #     南充/十二指肠/...
    output_dir = OUTPUT_ROOT / city / organ
    output_path = output_dir / OUTPUT_FILENAME_TEMPLATE.format(
        city=city,
        organ=organ,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(output_path)

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        k = record_query_key(r)
        if k:
            existing_by_key[k] = r

    print("=" * 110)
    print(f"[INPUT ] {source_json}")
    print(f"[CITY  ] {city}")
    print(f"[ORGAN ] {organ}")
    print(f"[OUTPUT] {output_path}")
    print(f"[N     ] {len(records)}")
    print(f"[RESUME] existing={len(existing_by_key)}")
    print("=" * 110)

    final_results: List[Dict[str, Any]] = []

    success_count = 0
    failure_count = 0

    for idx, record in enumerate(records):
        image_path = get_image_path(record)
        ground_truth = get_ground_truth(record)

        if not image_path:
            raise KeyError(
                f"record[{idx}] 缺少 target_image/image_path/query_path"
            )

        key = record_query_key(record)

        if (
            RESUME
            and not OVERWRITE_EXISTING_QUERY
            and key
            and key in existing_by_key
            and existing_by_key[key].get("mllm_service_success") is True
        ):
            result_record = existing_by_key[key]
            final_results.append(result_record)
            print(
                f"[{idx + 1:04d}/{len(records):04d}] "
                f"SKIP existing | {os.path.basename(image_path)}"
            )
            continue

        print(
            f"[{idx + 1:04d}/{len(records):04d}] "
            f"{os.path.basename(image_path)} | GT={ground_truth}"
        )

        try:
            support_classes = normalize_support_classes(record)

            response = client.infer(
                target_image=image_path,
                support_classes=support_classes,
            )

            diag = str(response.get("diag") or "未知")
            candidate_diseases = (
                response.get("candidate_diseases") or []
            )
            score_map = response.get("mllm_score_map") or {}

            # 检查 FULL-SET score map 完整性
            expected_classes = [
                x["label_name"] for x in support_classes
            ]
            missing_scores = [
                c for c in expected_classes
                if c not in score_map
            ]

            if missing_scores:
                raise KeyError(
                    "mllm_score_map 不是 full-set，缺失类别: "
                    + ", ".join(missing_scores[:20])
                )

            mllm_correct = (
                class_eq(diag, ground_truth)
                if ground_truth is not None
                else None
            )

            # 保留原始 agent record，最大限度兼容后续 paired alignment / GT。
            result_record = dict(record)

            result_record.update(
                {
                    # fusion 对齐最常用字段
                    "image_path": image_path,
                    "target_image": image_path,
                    "ground_truth": ground_truth,

                    # MLLM 主字段
                    "mllm_diag": diag,
                    "mllm_correct": mllm_correct,
                    "mllm_score_map": score_map,

                    # candidate 只做诊断记录，不控制 FSL
                    "candidate_diseases": candidate_diseases,
                    "candidate_support_dirs": (
                        response.get("candidate_support_dirs")
                        or []
                    ),
                    "mllm_candidates_detail": (
                        response.get("candidates")
                        or []
                    ),

                    # 其它 MLLM 信息
                    "mllm_visual_evidence": (
                        response.get("visual_evidence") or ""
                    ),
                    "mllm_risk_level": (
                        response.get("risk_level") or ""
                    ),
                    "mllm_raw_text": (
                        response.get("raw_text") or ""
                    ),

                    # runner/service diagnostics
                    "mllm_service_success": True,
                    "mllm_service_error": None,
                    "mllm_time_cost": float(
                        response.get("_time_cost") or 0.0
                    ),
                    "mllm_full_score_class_count": len(score_map),
                    "mllm_candidate_count": len(candidate_diseases),
                    "mllm_service_protocol": {
                        "candidate_diseases_used_for_fsl": False,
                        "fsl_source": "offline_full_set",
                        "score_map_scope": "full_support_classes",
                        "teacher_forcing": USE_TEACHER_FORCING,
                        "softmax_temperature": SCORE_SOFTMAX_TEMPERATURE,
                        "length_normalize": LENGTH_NORMALIZE,
                    },
                }
            )

            final_results.append(result_record)
            success_count += 1

            top_score = None
            if score_map:
                valid = [
                    (c, x.get("norm_prob"))
                    for c, x in score_map.items()
                    if isinstance(x, dict)
                    and x.get("norm_prob") is not None
                ]
                if valid:
                    top_score = max(
                        valid,
                        key=lambda x: float(x[1]),
                    )

            print(
                f"    diag={diag} | "
                f"correct={mllm_correct} | "
                f"candidate={candidate_diseases} | "
                f"score_top1={top_score}"
            )

        except Exception as exc:
            failure_count += 1
            traceback.print_exc()

            failure_record = dict(record)
            failure_record.update(
                {
                    "image_path": image_path,
                    "target_image": image_path,
                    "ground_truth": ground_truth,
                    "mllm_diag": "未知",
                    "mllm_correct": False if ground_truth is not None else None,
                    "mllm_score_map": {},
                    "candidate_diseases": [],
                    "candidate_support_dirs": [],
                    "mllm_service_success": False,
                    "mllm_service_error": repr(exc),
                }
            )
            final_results.append(failure_record)

            if not CONTINUE_ON_ERROR:
                doc = build_output_document(
                    source_json=source_json,
                    city=city,
                    organ=organ,
                    results=final_results,
                )
                atomic_save_json(doc, output_path)
                raise

        # 断点保存
        if (
            SAVE_EVERY > 0
            and len(final_results) % SAVE_EVERY == 0
        ):
            doc = build_output_document(
                source_json=source_json,
                city=city,
                organ=organ,
                results=final_results,
            )
            atomic_save_json(doc, output_path)

    # 最终保存
    doc = build_output_document(
        source_json=source_json,
        city=city,
        organ=organ,
        results=final_results,
    )
    atomic_save_json(doc, output_path)

    print(
        f"[DONE] city={city} | organ={organ} | "
        f"success={success_count} | failure={failure_count} | "
        f"saved={output_path}"
    )

    return output_path


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    matched = sorted(
        Path(p).resolve()
        for p in glob.glob(
            INPUT_GLOB,
            recursive=True,
        )
        if Path(p).is_file()
    )

    if not matched:
        raise FileNotFoundError(
            f"INPUT_GLOB 没有匹配到 JSON: {INPUT_GLOB}"
        )

    print("=" * 110)
    print("[MLLM Candidate Service Batch Runner / FULL score map]")
    print(f"endpoint = {MLLM_ENDPOINT}")
    print(f"input    = {INPUT_GLOB}")
    print(f"files    = {len(matched)}")
    print(f"output   = {OUTPUT_ROOT}")
    print("FSL      = NOT executed here; offline full-set FSL is used later")
    print("=" * 110)

    client = MLLMCandidateClient(
        endpoint=MLLM_ENDPOINT,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
        retry_sleep=RETRY_SLEEP_SECONDS,
    )

    health = client.healthcheck()
    print(f"[SERVICE OK] {health}")

    # 预检查：每个输入 JSON 必须对应唯一 city-organ，避免任何静默覆盖。
    destination_seen: Dict[Tuple[str, str], Path] = {}
    for source_json in matched:
        preview_records = extract_records(load_json(source_json))
        if not preview_records:
            raise ValueError(f"输入 JSON 没有 records: {source_json}")
        preview_city = infer_city_from_record_or_path(
            preview_records[0], source_json
        )
        preview_organ = infer_organ_from_record_or_path(
            preview_records[0], source_json
        )
        dataset_key = (preview_city, preview_organ)
        if dataset_key in destination_seen:
            raise RuntimeError(
                "发现两个输入 JSON 会写入同一个 city-organ 输出，已阻止覆盖：\n"
                f"  dataset={preview_city}-{preview_organ}\n"
                f"  first={destination_seen[dataset_key]}\n"
                f"  second={source_json}"
            )
        destination_seen[dataset_key] = source_json

    print("[DATASETS]")
    for (city, organ), src in sorted(destination_seen.items()):
        print(f"  - {city}-{organ}: {src}")
    print(f"[DATASET COUNT] {len(destination_seen)}")

    outputs = []
    for source_json in matched:
        try:
            outputs.append(
                process_json(
                    source_json=source_json,
                    client=client,
                )
            )
        except Exception as exc:
            print(
                f"[ERROR] failed file: {source_json} | {repr(exc)}"
            )
            traceback.print_exc()

            if not CONTINUE_ON_ERROR:
                raise

    print("=" * 110)
    print("[ALL DONE]")
    for p in outputs:
        print(f"  - {p}")
    print("=" * 110)


if __name__ == "__main__":
    main()
