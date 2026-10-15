#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build SFT cold-start data + held-out GRPO data for medical single-choice VQA.

One script handles:
  1) deterministic image-level stratified SFT/RL split
  2) stable image_id / sample_id / candidate_id
  3) Qwen3-VL-32B teacher generation with Transformers
  4) rule-based leakage / format checks
  5) Qwen3-VL-32B visual judge
  6) SQLite resume
  7) export sft_train.json / rl_train.json / manifests / statistics

Typical usage
-------------
# 1. First inspect the split without loading the 32B model:
python build_sft_rl_data.py \
  --input_json /path/to/medical_vqa_train_rl.json \
  --output_dir /path/to/data/cold_start_huaxi \
  --prepare_only

# 2. Generate + judge. Re-running this exact command automatically resumes:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python build_sft_rl_data.py \
  --input_json /path/to/medical_vqa_train_rl.json \
  --output_dir /path/to/data/cold_start_huaxi \
  --teacher_model /path/to/Qwen3-VL-32B-Instruct

# 3. Re-export files from SQLite without loading the model:
python build_sft_rl_data.py \
  --input_json /path/to/medical_vqa_train_rl.json \
  --output_dir /path/to/data/cold_start_huaxi \
  --export_only
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_SFT_RATIO = 0.30
DEFAULT_SEED = 42

# Generate several candidates, but retain at most one accepted response per
# original sample to avoid over-representing the same image/question.
DEFAULT_CANDIDATES_PER_ATTEMPT = 2
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_JUDGE_THRESHOLD = 0.80

DEFAULT_GEN_MAX_NEW_TOKENS = 320
DEFAULT_GEN_TEMPERATURE = 0.70
DEFAULT_GEN_TOP_P = 0.90

DEFAULT_JUDGE_MAX_NEW_TOKENS = 220

DB_NAME = "sft_build_state.sqlite3"
SPLIT_MANIFEST_NAME = "split_manifest.json"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SFT_OUTPUT_NAME = "sft_train.json"
RL_OUTPUT_NAME = "rl_train.json"
ACCEPTED_JSONL_NAME = "accepted.jsonl"
REJECTED_JSONL_NAME = "rejected.jsonl"
STATS_NAME = "statistics.json"

ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(option_[A-Z])\s*</answer>",
    flags=re.IGNORECASE,
)
ANY_OPTION_RE = re.compile(r"\boption_[A-Z]\b", flags=re.IGNORECASE)

FORBIDDEN_PHRASES = [
    "ground truth",
    "ground-truth",
    "reference answer",
    "reference label",
    "provided answer",
    "provided label",
    "target answer",
    "target label",
    "gold answer",
    "gold label",
    "correct answer is",
    "correct option is",
    "the answer is option_",
    "according to the label",
    "according to the answer",
    "given answer",
    "given label",
]

FORBIDDEN_TAGS = [
    "<reasoning>",
    "</reasoning>",
    "<analysis>",
    "</analysis>",
    "<crop>",
    "</crop>",
    "<tool",
    "</tool",
    "<toolresponse",
    "</toolresponse",
]


# =============================================================================
# Utility
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_text(text: str, n: int = 20) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def normalize_path_for_id(path: str) -> str:
    # Do not resolve() because files may live on another machine/mount.
    # Only normalize redundant separators and whitespace.
    return os.path.normpath(str(path).strip())


def normalize_option(value: Any) -> Optional[str]:
    if value is None:
        return None
    m = re.fullmatch(r"\s*option_([A-Z])\s*", str(value), flags=re.IGNORECASE)
    if not m:
        return None
    return f"option_{m.group(1).upper()}"


def get_user_text(item: Dict[str, Any]) -> str:
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    extra = item.get("extra_info", {})
    return str(extra.get("question", ""))


def get_image_path(item: Dict[str, Any]) -> str:
    images = item.get("images") or []
    if not images:
        raise ValueError("sample has no images")
    if len(images) != 1:
        raise ValueError(
            f"This classic VQA builder expects exactly one image per sample; got {len(images)}"
        )
    return str(images[0])


def get_reference_answer(item: Dict[str, Any]) -> str:
    ans = normalize_option(item.get("extra_info", {}).get("reference_answer"))
    if ans is None:
        raise ValueError("invalid/missing extra_info.reference_answer")
    return ans


def get_options(item: Dict[str, Any]) -> Dict[str, str]:
    raw = item.get("extra_info", {}).get("options")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("missing extra_info.options")
    out: Dict[str, str] = {}
    for k, v in raw.items():
        nk = normalize_option(k)
        if nk is None:
            continue
        out[nk] = str(v)
    if not out:
        raise ValueError("no valid option_X keys in extra_info.options")
    return out


def get_reference_disease(item: Dict[str, Any]) -> str:
    extra = item.get("extra_info", {})
    disease = extra.get("reference_disease")
    if disease:
        return str(disease)
    ref = get_reference_answer(item)
    options = get_options(item)
    if ref not in options:
        raise ValueError(f"{ref} absent from options")
    return options[ref]


def validate_source_item(item: Dict[str, Any], index: int) -> None:
    _ = get_user_text(item)
    _ = get_image_path(item)
    ref = get_reference_answer(item)
    options = get_options(item)
    disease = get_reference_disease(item)

    if ref not in options:
        raise ValueError(f"index={index}: reference_answer {ref} absent from options")
    if options[ref] != disease:
        raise ValueError(
            f"index={index}: label mismatch: options[{ref}]={options[ref]!r} "
            f"!= reference_disease={disease!r}"
        )


def make_ids(item: Dict[str, Any], index: int) -> Tuple[str, str]:
    image_path = normalize_path_for_id(get_image_path(item))
    image_id = "img_" + sha1_text(image_path, 20)

    # sample_id changes if question/options/label changes, but remains stable
    # across JSON reordering.
    payload = {
        "image_id": image_id,
        "question": get_user_text(item),
        "options": get_options(item),
        "reference_answer": get_reference_answer(item),
        "reference_disease": get_reference_disease(item),
    }
    sample_id = "sample_" + sha1_text(stable_json(payload), 20)
    return image_id, sample_id


def candidate_id(sample_id: str, attempt: int, candidate_no: int) -> str:
    return f"{sample_id}_a{attempt:02d}_c{candidate_no:02d}"


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("input JSON must be a top-level list")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# SQLite state
# =============================================================================

def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            image_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            split TEXT NOT NULL,
            disease TEXT NOT NULL,
            reference_answer TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            accepted_candidate_id TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_samples_split_status
            ON samples(split, status);
        CREATE INDEX IF NOT EXISTS idx_samples_image_id
            ON samples(image_id);

        CREATE TABLE IF NOT EXISTS generations (
            candidate_id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            candidate_no INTEGER NOT NULL,
            response TEXT NOT NULL,
            parsed_answer TEXT,
            rule_pass INTEGER NOT NULL,
            rule_reason TEXT NOT NULL,
            judge_pass INTEGER,
            judge_score REAL,
            judge_reason TEXT,
            judge_raw TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
        );

        CREATE INDEX IF NOT EXISTS idx_generations_sample
            ON generations(sample_id);
        """
    )
    return conn


def metadata_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def metadata_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO metadata(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# =============================================================================
# Deterministic image-level stratified split
# =============================================================================

def build_split_assignments(
    data: List[Dict[str, Any]],
    sft_ratio: float,
    seed: int,
    min_rl_images_per_class: int = 1,
) -> Dict[str, str]:
    """
    Returns image_id -> "sft"/"rl".

    Grouping is by image_id so the same source image can never cross SFT/RL,
    even when the JSON contains multiple option-order variants of that image.
    """
    disease_to_images: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for idx, item in enumerate(data):
        image_id, _ = make_ids(item, idx)
        disease = get_reference_disease(item)
        disease_to_images[disease][image_id].append(idx)

    rng = random.Random(seed)
    assignments: Dict[str, str] = {}

    for disease in sorted(disease_to_images):
        image_ids = sorted(disease_to_images[disease].keys())
        rng.shuffle(image_ids)
        n = len(image_ids)

        # Preserve at least min_rl_images_per_class in RL whenever possible.
        max_sft = max(0, n - min_rl_images_per_class)
        wanted = int(round(n * sft_ratio))

        if n <= min_rl_images_per_class:
            n_sft = 0
        else:
            # If the class is sufficiently represented, ensure at least one
            # image enters SFT, while never stealing the last RL image.
            n_sft = min(max_sft, max(1, wanted))

        sft_set = set(image_ids[:n_sft])
        for image_id in image_ids:
            assignments[image_id] = "sft" if image_id in sft_set else "rl"

    return assignments


def initialize_or_validate_state(
    conn: sqlite3.Connection,
    data: List[Dict[str, Any]],
    input_path: Path,
    output_dir: Path,
    sft_ratio: float,
    seed: int,
) -> None:
    source_sha = sha256_file(input_path)
    stored_sha = metadata_get(conn, "source_sha256")

    if stored_sha is not None and stored_sha != source_sha:
        raise RuntimeError(
            "Existing resume DB belongs to a different source JSON.\n"
            f"  db source_sha256 : {stored_sha}\n"
            f"  new source_sha256: {source_sha}\n"
            "Use a new --output_dir for the new training JSON."
        )

    if stored_sha is None:
        print("[INIT] validating source JSON...")
        seen_sample_ids = set()
        for idx, item in enumerate(data):
            validate_source_item(item, idx)
            _, sid = make_ids(item, idx)
            if sid in seen_sample_ids:
                # Exact duplicate sample is harmless, but a PK collision would
                # make source_index ambiguous. Fail loudly instead.
                raise ValueError(
                    f"duplicate stable sample_id detected at source index {idx}: {sid}"
                )
            seen_sample_ids.add(sid)

        assignments = build_split_assignments(
            data=data,
            sft_ratio=sft_ratio,
            seed=seed,
        )

        created = now_iso()
        for idx, item in enumerate(data):
            image_id, sid = make_ids(item, idx)
            split = assignments[image_id]
            disease = get_reference_disease(item)
            ref = get_reference_answer(item)
            status = "PENDING" if split == "sft" else "RL_READY"
            conn.execute(
                """
                INSERT INTO samples(
                    sample_id,image_id,source_index,split,disease,
                    reference_answer,status,attempt_count,
                    accepted_candidate_id,last_error,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sid, image_id, idx, split, disease, ref, status, 0,
                    None, None, created, created,
                ),
            )

        metadata_set(conn, "source_sha256", source_sha)
        metadata_set(conn, "source_path", str(input_path))
        metadata_set(conn, "sft_ratio", str(sft_ratio))
        metadata_set(conn, "seed", str(seed))
        metadata_set(conn, "created_at", created)
        conn.commit()

        write_source_and_split_manifests(conn, data, input_path, output_dir)
    else:
        old_ratio = float(metadata_get(conn, "sft_ratio") or sft_ratio)
        old_seed = int(metadata_get(conn, "seed") or seed)
        if abs(old_ratio - sft_ratio) > 1e-12 or old_seed != seed:
            print(
                "[RESUME] Existing split manifest wins. "
                f"Stored ratio/seed={old_ratio}/{old_seed}; "
                f"requested={sft_ratio}/{seed}."
            )


def write_source_and_split_manifests(
    conn: sqlite3.Connection,
    data: List[Dict[str, Any]],
    input_path: Path,
    output_dir: Path,
) -> None:
    rows = conn.execute(
        """
        SELECT sample_id,image_id,source_index,split,disease,reference_answer,status
        FROM samples ORDER BY source_index
        """
    ).fetchall()

    source_manifest = {
        "source_path": str(input_path),
        "source_sha256": metadata_get(conn, "source_sha256"),
        "num_samples": len(data),
        "created_at": metadata_get(conn, "created_at"),
    }
    write_json(output_dir / SOURCE_MANIFEST_NAME, source_manifest)

    split_manifest = {
        "seed": int(metadata_get(conn, "seed") or DEFAULT_SEED),
        "sft_ratio_requested": float(
            metadata_get(conn, "sft_ratio") or DEFAULT_SFT_RATIO
        ),
        "grouping": "image_id",
        "samples": [
            {
                "source_index": int(r["source_index"]),
                "sample_id": r["sample_id"],
                "image_id": r["image_id"],
                "split": r["split"],
                "reference_disease": r["disease"],
                "reference_answer": r["reference_answer"],
            }
            for r in rows
        ],
    }
    write_json(output_dir / SPLIT_MANIFEST_NAME, split_manifest)


# =============================================================================
# Teacher / Judge
# =============================================================================

class Qwen3VLTeacher:
    def __init__(
        self,
        model_path: str,
        attn_impl: str = "flash_attention_2",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_path = model_path

        print(f"[MODEL] loading teacher/judge: {model_path}")
        print(f"[MODEL] requested attention implementation: {attn_impl}")

        kwargs = dict(
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation=attn_impl,
                **kwargs,
            )
        except Exception as exc:
            if attn_impl == "sdpa":
                raise
            print(
                "[MODEL][WARN] flash_attention_2 load failed; "
                f"falling back to sdpa.\n  {type(exc).__name__}: {exc}"
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="sdpa",
                **kwargs,
            )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()

        # For sharded device_map="auto", inputs should enter the device holding
        # the input embeddings. This is more reliable than blindly using cuda:0.
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except Exception:
            self.input_device = next(self.model.parameters()).device

        print(f"[MODEL] input device: {self.input_device}")

    def _generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        from PIL import Image

        with Image.open(image_path) as im:
            image = im.convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)

            moved = {}
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    moved[k] = v.to(self.input_device)
                else:
                    moved[k] = v
            inputs = moved

            generation_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            if do_sample:
                generation_kwargs.update(
                    temperature=temperature,
                    top_p=top_p,
                )

            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_kwargs)

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            text = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return text.strip()

    def generate_candidate(
        self,
        item: Dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        question = get_user_text(item)
        ref = get_reference_answer(item)
        disease = get_reference_disease(item)
        options = get_options(item)

        # IMPORTANT: never serialize extra_info or image_path into this prompt.
        # The target is deliberately supplied only to constrain the SFT label.
        prompt = f"""
You are creating one high-quality supervised fine-tuning response for a
gastrointestinal endoscopy visual single-choice diagnosis task.

QUESTION:
{question}

TARGET (private supervision only; NEVER mention that it was provided):
- target option: {ref}
- target diagnosis: {disease}

Candidate mapping for consistency:
{stable_json(options)}

Write the assistant response that a competent model should produce after
independently examining the image.

Requirements:
1. Use natural prose only. Do NOT use <reasoning> or <analysis> tags.
2. Give a short, image-grounded diagnostic explanation, usually 2-4 sentences.
3. Mention concrete visible endoscopic findings when they are actually
   supportable from the image.
4. You may briefly distinguish the most relevant alternative diagnosis.
5. Do NOT say or imply that you were given the target, label, answer, ground
   truth, reference answer, or correct option.
6. In the prose before the final tag, do NOT write option_A/option_B/etc.
7. Do NOT mention file names, directories, image paths, metadata, or hidden
   fields.
8. Do NOT use crop/tool/observe actions.
9. End with exactly one final tag and nothing after it:
   <answer>{ref}</answer>

Return only the assistant response.
""".strip()

        return self._generate(
            image_path=get_image_path(item),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    def judge_candidate(
        self,
        item: Dict[str, Any],
        response: str,
        max_new_tokens: int,
    ) -> Tuple[bool, float, str, str]:
        question = get_user_text(item)
        ref = get_reference_answer(item)
        disease = get_reference_disease(item)

        prompt = f"""
Act as a strict quality-control judge for a medical VQA SFT target.

You can see the endoscopy image. Evaluate whether the proposed assistant
response is suitable as supervised training data.

QUESTION:
{question}

REFERENCE FOR QC ONLY:
- expected final option: {ref}
- expected diagnosis: {disease}

PROPOSED RESPONSE:
---BEGIN RESPONSE---
{response}
---END RESPONSE---

Judge these properties:
- final answer is medically and label-wise consistent with the reference;
- explanation is plausibly grounded in visible image findings rather than
  generic filler;
- explanation does not reveal or imply access to a supplied answer/label;
- no fabricated metadata/path/tool/crop/hidden information;
- no major contradiction between explanation and diagnosis;
- concise enough for cold-start SFT.

Return ONLY one JSON object with exactly these keys:
{{
  "pass": true,
  "visual_grounding": true,
  "no_leakage": true,
  "medical_consistency": true,
  "concise": true,
  "score": 0.95,
  "reason": "short reason"
}}

Set "pass" to true only if all four boolean quality fields are true.
score must be between 0 and 1.
""".strip()

        raw = self._generate(
            image_path=get_image_path(item),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        obj = extract_json_object(raw)
        if obj is None:
            return False, 0.0, "judge_json_parse_failed", raw

        try:
            visual = bool(obj["visual_grounding"])
            no_leak = bool(obj["no_leakage"])
            med = bool(obj["medical_consistency"])
            concise = bool(obj["concise"])
            passed = bool(obj["pass"]) and visual and no_leak and med and concise
            score = float(obj["score"])
            score = max(0.0, min(1.0, score))
            reason = str(obj.get("reason", ""))
        except Exception as exc:
            return False, 0.0, f"judge_schema_error:{exc}", raw

        return passed, score, reason, raw


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # Remove common markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Best-effort first balanced outer object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# =============================================================================
# Rule checks
# =============================================================================

def rule_check_response(
    response: str,
    reference_answer: str,
) -> Tuple[bool, str, Optional[str]]:
    if not isinstance(response, str) or not response.strip():
        return False, "empty_response", None

    text = response.strip()
    matches = ANSWER_TAG_RE.findall(text)

    if len(matches) != 1:
        return False, f"answer_tag_count={len(matches)}", None

    parsed = normalize_option(f"option_{matches[0].split('_')[-1]}")
    if parsed is None:
        return False, "invalid_answer_option", None

    if parsed != reference_answer:
        return False, f"wrong_answer:{parsed}!={reference_answer}", parsed

    # Final answer tag must be the literal ending.
    final_pattern = re.compile(
        rf"<answer>\s*{re.escape(reference_answer)}\s*</answer>\s*$",
        flags=re.IGNORECASE,
    )
    if not final_pattern.search(text):
        return False, "answer_tag_not_final", parsed

    # Everything before the final answer.
    body = final_pattern.sub("", text).strip()

    # Cold-start target should contain some explanatory content, not only GT.
    if len(body) < 40:
        return False, f"explanation_too_short:{len(body)}", parsed

    lower = body.lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return False, f"forbidden_phrase:{phrase}", parsed

    for tag in FORBIDDEN_TAGS:
        if tag in lower:
            return False, f"forbidden_tag:{tag}", parsed

    # Do not let the prose announce option_X before the final answer tag.
    body_options = ANY_OPTION_RE.findall(body)
    if body_options:
        return False, f"option_label_in_explanation:{body_options[0]}", parsed

    # Prevent accidental path/metadata regurgitation.
    if re.search(r"\.(?:jpg|jpeg|png|webp)\b", body, flags=re.IGNORECASE):
        return False, "image_filename_leak", parsed
    if "/mnt/" in body or "\\mnt\\" in body:
        return False, "image_path_leak", parsed

    return True, "ok", parsed


# =============================================================================
# Generation loop + resume
# =============================================================================

def process_sft_samples(
    conn: sqlite3.Connection,
    data: List[Dict[str, Any]],
    teacher: Qwen3VLTeacher,
    output_dir: Path,
    candidates_per_attempt: int,
    max_attempts: int,
    judge_threshold: float,
    gen_max_new_tokens: int,
    gen_temperature: float,
    gen_top_p: float,
    judge_max_new_tokens: int,
) -> None:
    rows = conn.execute(
        """
        SELECT * FROM samples
        WHERE split='sft' AND status NOT IN ('ACCEPTED','EXHAUSTED')
        ORDER BY source_index
        """
    ).fetchall()

    total_sft = conn.execute(
        "SELECT COUNT(*) AS n FROM samples WHERE split='sft'"
    ).fetchone()["n"]
    accepted_before = conn.execute(
        "SELECT COUNT(*) AS n FROM samples WHERE split='sft' AND status='ACCEPTED'"
    ).fetchone()["n"]

    print(
        f"[RESUME] SFT total={total_sft}, accepted={accepted_before}, "
        f"remaining={len(rows)}"
    )

    accepted_jsonl = output_dir / ACCEPTED_JSONL_NAME
    rejected_jsonl = output_dir / REJECTED_JSONL_NAME

    for pos, row in enumerate(rows, start=1):
        sid = str(row["sample_id"])
        src_idx = int(row["source_index"])
        item = data[src_idx]
        ref = get_reference_answer(item)

        current_attempt = int(row["attempt_count"])
        already_accepted = row["accepted_candidate_id"] is not None

        if already_accepted:
            continue

        print(
            f"\n[{pos}/{len(rows)}] {sid} "
            f"disease={row['disease']} attempts={current_attempt}/{max_attempts}"
        )

        accepted_cid: Optional[str] = None

        while current_attempt < max_attempts and accepted_cid is None:
            current_attempt += 1

            conn.execute(
                "UPDATE samples SET status=?,attempt_count=?,updated_at=? "
                "WHERE sample_id=?",
                ("GENERATING", current_attempt, now_iso(), sid),
            )
            conn.commit()

            # If generation rows from this attempt already exist (e.g. crash
            # after generation but before sample status update), reuse them.
            existing = conn.execute(
                """
                SELECT * FROM generations
                WHERE sample_id=? AND attempt=?
                ORDER BY candidate_no
                """,
                (sid, current_attempt),
            ).fetchall()

            existing_by_no = {int(r["candidate_no"]): r for r in existing}

            for cno in range(1, candidates_per_attempt + 1):
                cid = candidate_id(sid, current_attempt, cno)
                gen_row = existing_by_no.get(cno)

                if gen_row is None:
                    try:
                        response = teacher.generate_candidate(
                            item=item,
                            max_new_tokens=gen_max_new_tokens,
                            temperature=gen_temperature,
                            top_p=gen_top_p,
                        )
                    except Exception as exc:
                        err = f"generation_error:{type(exc).__name__}:{exc}"
                        print(f"  [GEN][ERROR] {cid}: {err}")
                        conn.execute(
                            "UPDATE samples SET last_error=?,updated_at=? "
                            "WHERE sample_id=?",
                            (err, now_iso(), sid),
                        )
                        conn.commit()
                        continue

                    rule_pass, rule_reason, parsed = rule_check_response(
                        response,
                        reference_answer=ref,
                    )

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO generations(
                            candidate_id,sample_id,attempt,candidate_no,response,
                            parsed_answer,rule_pass,rule_reason,judge_pass,
                            judge_score,judge_reason,judge_raw,created_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            cid, sid, current_attempt, cno, response, parsed,
                            int(rule_pass), rule_reason, None, None, None, None,
                            now_iso(),
                        ),
                    )
                    conn.commit()

                    print(
                        f"  [RULE] {cid}: "
                        f"{'PASS' if rule_pass else 'REJECT'} ({rule_reason})"
                    )

                    if not rule_pass:
                        append_jsonl(
                            rejected_jsonl,
                            {
                                "candidate_id": cid,
                                "sample_id": sid,
                                "stage": "rule",
                                "reason": rule_reason,
                                "response": response,
                            },
                        )
                        continue
                else:
                    response = str(gen_row["response"])
                    rule_pass = bool(gen_row["rule_pass"])
                    rule_reason = str(gen_row["rule_reason"])
                    print(
                        f"  [RESUME] reuse {cid}: "
                        f"rule={'PASS' if rule_pass else 'REJECT'}"
                    )
                    if not rule_pass:
                        continue

                # See if judge was already completed before an interruption.
                gen_row2 = conn.execute(
                    "SELECT * FROM generations WHERE candidate_id=?",
                    (cid,),
                ).fetchone()

                if gen_row2["judge_pass"] is None:
                    try:
                        judge_pass, score, reason, raw = teacher.judge_candidate(
                            item=item,
                            response=response,
                            max_new_tokens=judge_max_new_tokens,
                        )
                    except Exception as exc:
                        judge_pass = False
                        score = 0.0
                        reason = f"judge_error:{type(exc).__name__}:{exc}"
                        raw = ""

                    effective_pass = bool(judge_pass and score >= judge_threshold)

                    conn.execute(
                        """
                        UPDATE generations SET
                            judge_pass=?,judge_score=?,judge_reason=?,judge_raw=?
                        WHERE candidate_id=?
                        """,
                        (
                            int(effective_pass),
                            float(score),
                            str(reason),
                            raw,
                            cid,
                        ),
                    )
                    conn.commit()
                else:
                    effective_pass = bool(gen_row2["judge_pass"])
                    score = float(gen_row2["judge_score"] or 0.0)
                    reason = str(gen_row2["judge_reason"] or "")

                print(
                    f"  [JUDGE] {cid}: "
                    f"{'PASS' if effective_pass else 'REJECT'} "
                    f"score={score:.3f} ({reason})"
                )

                if effective_pass:
                    accepted_cid = cid
                    conn.execute(
                        """
                        UPDATE samples
                        SET status='ACCEPTED',
                            accepted_candidate_id=?,
                            last_error=NULL,
                            updated_at=?
                        WHERE sample_id=?
                        """,
                        (cid, now_iso(), sid),
                    )
                    conn.commit()

                    append_jsonl(
                        accepted_jsonl,
                        {
                            "candidate_id": cid,
                            "sample_id": sid,
                            "source_index": src_idx,
                            "judge_score": score,
                            "response": response,
                        },
                    )
                    break
                else:
                    append_jsonl(
                        rejected_jsonl,
                        {
                            "candidate_id": cid,
                            "sample_id": sid,
                            "stage": "judge",
                            "reason": reason,
                            "judge_score": score,
                            "response": response,
                        },
                    )

        if accepted_cid is None:
            conn.execute(
                """
                UPDATE samples
                SET status='EXHAUSTED',updated_at=?
                WHERE sample_id=?
                """,
                (now_iso(), sid),
            )
            conn.commit()
            print(f"  [EXHAUSTED] {sid}")

        # Export incrementally. If the machine dies, accepted examples already
        # exist in both SQLite and the current JSON export.
        export_outputs(conn, data, output_dir)


# =============================================================================
# Export
# =============================================================================

def export_outputs(
    conn: sqlite3.Connection,
    data: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    sft_rows = conn.execute(
        """
        SELECT s.*, g.response, g.judge_score
        FROM samples s
        JOIN generations g ON s.accepted_candidate_id=g.candidate_id
        WHERE s.split='sft' AND s.status='ACCEPTED'
        ORDER BY s.source_index
        """
    ).fetchall()

    rl_rows = conn.execute(
        """
        SELECT * FROM samples
        WHERE split='rl'
        ORDER BY source_index
        """
    ).fetchall()

    sft_data: List[Dict[str, Any]] = []
    for row in sft_rows:
        item = copy.deepcopy(data[int(row["source_index"])])
        messages = list(item.get("messages", []))

        # Remove any pre-existing assistant message defensively.
        messages = [m for m in messages if m.get("role") != "assistant"]
        messages.append(
            {
                "role": "assistant",
                "content": str(row["response"]).strip(),
            }
        )
        item["messages"] = messages

        extra = dict(item.get("extra_info", {}))
        extra["cold_start_sample_id"] = row["sample_id"]
        extra["cold_start_image_id"] = row["image_id"]
        extra["cold_start_candidate_id"] = row["accepted_candidate_id"]
        extra["cold_start_judge_score"] = float(row["judge_score"] or 0.0)
        item["extra_info"] = extra

        sft_data.append(item)

    rl_data = [
        copy.deepcopy(data[int(row["source_index"])])
        for row in rl_rows
    ]

    write_json(output_dir / SFT_OUTPUT_NAME, sft_data)
    write_json(output_dir / RL_OUTPUT_NAME, rl_data)
    write_source_and_split_manifests(
        conn,
        data,
        Path(metadata_get(conn, "source_path") or ""),
        output_dir,
    )

    status_counts = {
        r["status"]: int(r["n"])
        for r in conn.execute(
            "SELECT status,COUNT(*) AS n FROM samples GROUP BY status"
        ).fetchall()
    }
    split_counts = {
        r["split"]: int(r["n"])
        for r in conn.execute(
            "SELECT split,COUNT(*) AS n FROM samples GROUP BY split"
        ).fetchall()
    }

    disease_rows = conn.execute(
        """
        SELECT disease,split,COUNT(*) AS n
        FROM samples
        GROUP BY disease,split
        ORDER BY disease,split
        """
    ).fetchall()
    by_disease: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in disease_rows:
        by_disease[str(r["disease"])][str(r["split"])] = int(r["n"])

    accepted_scores = [
        float(r["judge_score"])
        for r in conn.execute(
            """
            SELECT g.judge_score
            FROM samples s
            JOIN generations g ON s.accepted_candidate_id=g.candidate_id
            WHERE s.status='ACCEPTED' AND g.judge_score IS NOT NULL
            """
        ).fetchall()
    ]

    stats = {
        "updated_at": now_iso(),
        "source_sha256": metadata_get(conn, "source_sha256"),
        "split_counts": split_counts,
        "status_counts": status_counts,
        "exported_sft_samples": len(sft_data),
        "exported_rl_samples": len(rl_data),
        "accepted_judge_score_mean": (
            sum(accepted_scores) / len(accepted_scores)
            if accepted_scores else None
        ),
        "by_disease": by_disease,
    }
    write_json(output_dir / STATS_NAME, stats)


def print_summary(conn: sqlite3.Connection, output_dir: Path) -> None:
    rows = conn.execute(
        """
        SELECT split,status,COUNT(*) AS n
        FROM samples
        GROUP BY split,status
        ORDER BY split,status
        """
    ).fetchall()
    print("\n" + "=" * 76)
    print("[STATE SUMMARY]")
    for r in rows:
        print(f"{r['split']:>4} | {r['status']:<12} | {int(r['n']):6d}")
    print(f"DB     : {output_dir / DB_NAME}")
    print(f"SFT    : {output_dir / SFT_OUTPUT_NAME}")
    print(f"RL     : {output_dir / RL_OUTPUT_NAME}")
    print(f"STATS  : {output_dir / STATS_NAME}")
    print("=" * 76)


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument(
        "--teacher_model",
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Local path or HF model id for Qwen3-VL-32B-Instruct.",
    )
    p.add_argument("--sft_ratio", type=float, default=DEFAULT_SFT_RATIO)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument(
        "--candidates_per_attempt",
        type=int,
        default=DEFAULT_CANDIDATES_PER_ATTEMPT,
    )
    p.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    p.add_argument(
        "--judge_threshold",
        type=float,
        default=DEFAULT_JUDGE_THRESHOLD,
    )

    p.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=DEFAULT_GEN_MAX_NEW_TOKENS,
    )
    p.add_argument(
        "--gen_temperature",
        type=float,
        default=DEFAULT_GEN_TEMPERATURE,
    )
    p.add_argument("--gen_top_p", type=float, default=DEFAULT_GEN_TOP_P)
    p.add_argument(
        "--judge_max_new_tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_NEW_TOKENS,
    )

    p.add_argument(
        "--attn_impl",
        choices=["flash_attention_2", "sdpa"],
        default="flash_attention_2",
    )

    p.add_argument(
        "--prepare_only",
        action="store_true",
        help="Build/validate stable IDs + split + SQLite, but do not load 32B.",
    )
    p.add_argument(
        "--export_only",
        action="store_true",
        help="Only regenerate output JSON/statistics from existing SQLite state.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if not (0.0 < args.sft_ratio < 1.0):
        raise ValueError("--sft_ratio must be between 0 and 1")
    if args.candidates_per_attempt < 1:
        raise ValueError("--candidates_per_attempt must be >=1")
    if args.max_attempts < 1:
        raise ValueError("--max_attempts must be >=1")

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    print(f"[DATA] loaded {len(data)} samples from {input_path}")

    conn = connect_db(output_dir / DB_NAME)
    initialize_or_validate_state(
        conn=conn,
        data=data,
        input_path=input_path,
        output_dir=output_dir,
        sft_ratio=args.sft_ratio,
        seed=args.seed,
    )

    export_outputs(conn, data, output_dir)
    print_summary(conn, output_dir)

    if args.prepare_only or args.export_only:
        return

    teacher = Qwen3VLTeacher(
        model_path=args.teacher_model,
        attn_impl=args.attn_impl,
    )

    process_sft_samples(
        conn=conn,
        data=data,
        teacher=teacher,
        output_dir=output_dir,
        candidates_per_attempt=args.candidates_per_attempt,
        max_attempts=args.max_attempts,
        judge_threshold=args.judge_threshold,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )

    export_outputs(conn, data, output_dir)
    print_summary(conn, output_dir)


if __name__ == "__main__":
    main()
