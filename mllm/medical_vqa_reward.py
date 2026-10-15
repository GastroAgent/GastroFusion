#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classic single-turn medical VQA reward plugin for ms-swift GRPO.

No crop.
No observe/tool loop.
No MultiTurnScheduler.

Reward = accuracy_weight * exact_option_accuracy
       + format_weight   * final_answer_format
"""

from __future__ import annotations

import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional

try:
    import wandb
except ImportError:
    wandb = None

from swift.rewards.orm import ORM, orms


ANSWER_RE = re.compile(
    r"<answer>\s*(option_[A-Z])\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
OPTION_RE = re.compile(r"^option_([A-Z])$", flags=re.IGNORECASE)


def _completion_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    # Defensive support for OpenAI-like response/message objects.
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content

    message = getattr(value, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content

    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        msg = value.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]

    return str(value)


def normalize_option(value: Any) -> Optional[str]:
    if value is None:
        return None
    m = OPTION_RE.fullmatch(str(value).strip())
    if m is None:
        return None
    return f"option_{m.group(1).upper()}"


def extract_answer(text: Any) -> Optional[str]:
    text = _completion_text(text)
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return normalize_option(matches[-1])


def final_format_score(text: Any) -> float:
    """
    Valid format:
      - exactly one <answer>option_X</answer>
      - answer tag is the end of the completion
      - option label is valid
      - no crop/tool tags

    Natural explanation before the final answer is allowed.
    """
    text = _completion_text(text).strip()

    if "<crop>" in text.lower() or "<tool" in text.lower():
        return 0.0

    matches = ANSWER_RE.findall(text)
    if len(matches) != 1:
        return 0.0

    option = normalize_option(matches[0])
    if option is None:
        return 0.0

    if re.search(
        rf"<answer>\s*{re.escape(option)}\s*</answer>\s*$",
        text,
        flags=re.IGNORECASE,
    ) is None:
        return 0.0

    return 1.0


class MedicalVQAReward(ORM):
    def __init__(self, args, **kwargs):
        super().__init__(args)

        # Keep total maximum reward at 2.0, matching the previous scale
        # (1.8 accuracy + 0.2 format), but remove the unrelated tool reward.
        self.acc_weight = float(os.getenv("ACC_REWARD_WEIGHT", "1.8"))
        self.format_weight = float(os.getenv("FORMAT_REWARD_WEIGHT", "0.2"))

        self.log_reward_components = str(
            os.getenv("LOG_REWARD_COMPONENTS", "1")
        ).lower() not in {"0", "false", "no"}

        self.reward_log_interval = int(
            os.getenv("REWARD_LOG_INTERVAL", "1")
        )
        self._reward_call_count = 0

    def __call__(
        self,
        completions,
        extra_info,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        acc_scores: List[float] = []
        format_scores: List[float] = []

        for completion, info in zip(completions, extra_info):
            reference = normalize_option(
                (info or {}).get("reference_answer")
            )
            predicted = extract_answer(completion)

            acc = float(
                reference is not None
                and predicted is not None
                and predicted == reference
            )
            fmt = final_format_score(completion)

            total = self.acc_weight * acc + self.format_weight * fmt

            acc_scores.append(acc)
            format_scores.append(fmt)
            rewards.append(total)

        self._log(
            acc_scores=acc_scores,
            format_scores=format_scores,
            total_rewards=rewards,
        )
        return rewards

    @staticmethod
    def _rank0() -> bool:
        return int(os.environ.get("RANK", "0")) == 0

    def _log(
        self,
        acc_scores: List[float],
        format_scores: List[float],
        total_rewards: List[float],
    ) -> None:
        if not self.log_reward_components:
            return
        if wandb is None or wandb.run is None or not self._rank0():
            return

        self._reward_call_count += 1
        if self._reward_call_count % self.reward_log_interval != 0:
            return

        acc_mean = mean(acc_scores) if acc_scores else 0.0
        fmt_mean = mean(format_scores) if format_scores else 0.0
        total_mean = mean(total_rewards) if total_rewards else 0.0

        wandb.log(
            {
                "reward_components/acc_reward": acc_mean,
                "reward_components/format_reward": fmt_mean,
                "reward_components/weighted_acc_reward":
                    self.acc_weight * acc_mean,
                "reward_components/weighted_format_reward":
                    self.format_weight * fmt_mean,
                "reward_components/total_reward": total_mean,
            },
            commit=False,
        )


orms["medical_vqa_reward"] = MedicalVQAReward