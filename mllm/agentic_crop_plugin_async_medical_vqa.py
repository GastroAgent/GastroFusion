from typing import Union, List
import asyncio
import ast
import base64
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean

try:
    import wandb
except ImportError:
    wandb = None

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    from openai import OpenAI

    AsyncOpenAI = None
from PIL import Image

from swift.rewards.orm import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from prompt import AgentPrompt
from qwen_vl_utils import fetch_image


def _extract_tag(text: Any, tag: str) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.findall(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    return match[-1].strip()


def extract_answer(text: str) -> Optional[str]:
    return _extract_tag(text, "answer")


def extract_crop(text: str) -> Optional[str]:
    return _extract_tag(text, "crop")


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)
    return message if isinstance(message, str) else str(message)


def load_pil_image(img: Any) -> Any:
    try:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, dict) and "bytes" in img:
            return Image.open(io.BytesIO(img["bytes"]))
        if isinstance(img, str):
            if os.path.exists(img):
                return Image.open(img)
            raw = img.split(",", 1)[-1] if "," in img else img
            return Image.open(io.BytesIO(base64.b64decode(raw)))
        if isinstance(img, bytes):
            return Image.open(io.BytesIO(img))
        if hasattr(img, "read"):
            return Image.open(img)
        return img
    except Exception:
        return img


Number = Union[int, float]


def denormalize_bbox(
    image_width: Number,
    image_height: Number,
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
    clip: bool = True,
    to_int: bool = True
) -> List[Number]:
    """
    将千分位归一化坐标反归一化为原图绝对像素坐标。

    参数:
        image_width: 原图宽度
        image_height: 原图高度
        x1, y1, x2, y2: 千分位归一化坐标，通常范围为 0~1000
        clip: 是否先将输入裁剪到 0~1000 范围
        to_int: 是否将输出四舍五入为整数

    返回:
        [abs_x1, abs_y1, abs_x2, abs_y2]
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须大于 0")

    if clip:
        x1 = max(0, min(x1, 1000))
        y1 = max(0, min(y1, 1000))
        x2 = max(0, min(x2, 1000))
        y2 = max(0, min(y2, 1000))

    abs_x1 = x1 / 1000.0 * image_width
    abs_y1 = y1 / 1000.0 * image_height
    abs_x2 = x2 / 1000.0 * image_width
    abs_y2 = y2 / 1000.0 * image_height

    result = [abs_x1, abs_y1, abs_x2, abs_y2]

    if to_int:
        result = [int(round(v)) for v in result]

    return result


class DocCropScheduler(MultiTurnScheduler):
    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        super().__init__(infer_engine, max_turns, *args, **kwargs)

    def check_finished(self, infer_request, response_choice, current_turn):
        should_stop = super().check_finished(infer_request, response_choice, current_turn)
        if should_stop:
            return True

        response = response_choice.message.content

        answer_tag = extract_answer(response)

        if answer_tag is not None:
            return True
        # 非法格式：停止并交由 reward 扣分
        return False

    def step(self, infer_request, response_choice, current_turn):
        # print(f"[INFO] current_turn: {current_turn}")
        # print(f"[INFO] infer_request: {infer_request}")
        question = infer_request.data_dict['extra_info']['question']
        infer_messages = infer_request.messages
        response = response_choice.message.content
        model_answer = extract_answer(response)
        crop_bbox = extract_crop(response)
        if model_answer is not None:
            return {"infer_request": infer_request}
        if crop_bbox is not None:
            # NOTE: this function is only compatible with the QwenVL series models
            # If you use another MLLM, please adjust the fetch_image function accordingly
            # ensure the returned img is of type PIL.Image.Image and
            # has been processed to a maximum size of max_pixels
            img = fetch_image(
                {'image': load_pil_image(infer_request.images[0])})
            try:
                crop_bbox = ast.literal_eval(crop_bbox)
                x1, y1, x2, y2 = crop_bbox
                x1, y1, x2, y2 = denormalize_bbox(
                    infer_request.data_dict['extra_info']['image_width'],
                    infer_request.data_dict['extra_info']['image_height'],
                    x1, y1, x2, y2,
                    clip=True,
                    to_int=True
                )
            except Exception:
                infer_request.messages.append(
                    {"role": "user", "content": f"{AgentPrompt.CONTINUE_PROMPT.format(question=question)}"})
                return {"infer_request": infer_request}
            if not self.validate_bbox(x1, y1, x2, y2):
                infer_request.messages.append(
                    {"role": "user", "content": f"{AgentPrompt.CONTINUE_PROMPT.format(question=question)}"})
                return {"infer_request": infer_request}
            crop_image = img.crop((x1, y1, x2, y2))

            new_images = list(infer_request.images)
            new_images.append(crop_image)
            infer_request.images = new_images

            infer_request.messages.append(
                {"role": "user", "content": f"{AgentPrompt.AFTER_CROP_PROMPT.format(question=question)}"})
            return {
                "infer_request": infer_request,
                "rollout_infos": {
                    "images": new_images
                }
            }
        if crop_bbox is None and model_answer is None:
            infer_request.messages.append(
                {"role": "user", "content": f"{AgentPrompt.CONTINUE_PROMPT.format(question=question)}"})
            return {"infer_request": infer_request}

        infer_request.messages.append(
            {"role": "user", "content": AgentPrompt.CONTINUE_PROMPT.format(
                question=question)}
        )
        return {"infer_request": infer_request}

    def validate_bbox(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False

        # 你可以按需要改成 < 2 或 <= 2
        if min(width, height) <= 2:
            return False

        return True

    def maybe_resize_bbox(self, bbox: List[int], origin_width: int, origin_height: int) -> List[int]:
        left, top, right, bottom = bbox
        left = max(0, min(left, origin_width - 1))
        top = max(0, min(top, origin_height - 1))
        right = max(1, min(right, origin_width))
        bottom = max(1, min(bottom, origin_height))
        self.validate_bbox(left, top, right, bottom)

        width = right - left
        height = bottom - top
        if min(width, height) >= 28:
            return [left, top, right, bottom]

        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = 28 / max(1, min(width, height))
        half_w = ceil(width * ratio * 0.5)
        half_h = ceil(height * ratio * 0.5)

        new_left = max(0, floor(center_x - half_w))
        new_top = max(0, floor(center_y - half_h))
        new_right = min(origin_width, ceil(center_x + half_w))
        new_bottom = min(origin_height, ceil(center_y + half_h))
        self.validate_bbox(new_left, new_top, new_right, new_bottom)
        return [new_left, new_top, new_right, new_bottom]



class DocCropReward(ORM):
    """
    Reward for single-choice medical VQA.

    Expected answer format:
        <answer>option_A</answer>

    The reference answer in extra_info["reference_answer"] should also be
    option_A / option_B / ... . Because this is a closed-set classification
    task, accuracy is computed by exact option matching rather than LLM-as-judge.
    """

    _OPTION_RE = re.compile(r"^option_([A-Z])$", flags=re.IGNORECASE)

    def __init__(self, args, **kwargs):
        super().__init__(args)

        self.log_reward_components = str(
            os.getenv("LOG_REWARD_COMPONENTS", "1")
        ).lower() not in ["0", "false", "no"]

        self.reward_log_interval = int(
            os.getenv("REWARD_LOG_INTERVAL", "1")
        )

        self._reward_call_count = 0

        # Keep the original reward weights so existing training configs do not
        # silently change scale.
        self.acc_weight = float(os.getenv("ACC_REWARD_WEIGHT", "1.8"))
        self.format_weight = float(os.getenv("FORMAT_REWARD_WEIGHT", "0.2"))
        self.tool_weight = float(os.getenv("TOOL_REWARD_WEIGHT", "1.0"))

    @classmethod
    def normalize_option(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        match = cls._OPTION_RE.fullmatch(value)
        if match is None:
            return None
        return f"option_{match.group(1).upper()}"

    def __call__(self, completions, extra_info, **kwargs) -> List[float]:
        rewards: List[float] = []
        messages = kwargs.get("messages") or [[] for _ in completions]

        acc_scores: List[float] = []
        format_scores: List[float] = []
        tool_scores: List[float] = []

        for completion, message, info in zip(completions, messages, extra_info):
            reference_answer = self.normalize_option(
                info.get("reference_answer")
            )
            model_answer = self.normalize_option(
                extract_answer(completion)
            )

            # Closed-set MCQ: exact option match.
            acc_score = float(
                reference_answer is not None
                and model_answer is not None
                and model_answer == reference_answer
            )

            format_score = self.cal_format_score(message)
            tool_score = self.cal_tool_score(message)

            reward = (
                self.acc_weight * acc_score
                + self.format_weight * format_score
                + self.tool_weight * tool_score
            )

            acc_scores.append(acc_score)
            format_scores.append(format_score)
            tool_scores.append(tool_score)
            rewards.append(reward)

        self._log_reward_components_to_wandb(
            acc_scores=acc_scores,
            format_scores=format_scores,
            tool_scores=tool_scores,
            total_rewards=rewards,
        )

        return rewards

    def _is_rank0(self) -> bool:
        rank = int(os.environ.get("RANK", "0"))
        return rank == 0

    def _log_reward_components_to_wandb(
        self,
        acc_scores: List[float],
        format_scores: List[float],
        tool_scores: List[float],
        total_rewards: List[float],
    ) -> None:
        if not self.log_reward_components:
            return
        if wandb is None:
            return
        if not self._is_rank0():
            return
        if wandb.run is None:
            return

        self._reward_call_count += 1
        if self._reward_call_count % self.reward_log_interval != 0:
            return

        acc_mean = mean(acc_scores) if acc_scores else 0.0
        format_mean = mean(format_scores) if format_scores else 0.0
        tool_mean = mean(tool_scores) if tool_scores else 0.0
        total_mean = mean(total_rewards) if total_rewards else 0.0

        wandb.log(
            {
                "reward_components/acc_reward": acc_mean,
                "reward_components/format_reward": format_mean,
                "reward_components/tool_reward": tool_mean,

                "reward_components/weighted_acc_reward":
                    self.acc_weight * acc_mean,
                "reward_components/weighted_format_reward":
                    self.format_weight * format_mean,
                "reward_components/weighted_tool_reward":
                    self.tool_weight * tool_mean,

                "reward_components/total_reward": total_mean,

                "reward_components/acc_reward_min":
                    min(acc_scores) if acc_scores else 0.0,
                "reward_components/acc_reward_max":
                    max(acc_scores) if acc_scores else 0.0,
                "reward_components/format_reward_min":
                    min(format_scores) if format_scores else 0.0,
                "reward_components/format_reward_max":
                    max(format_scores) if format_scores else 0.0,
                "reward_components/tool_reward_min":
                    min(tool_scores) if tool_scores else 0.0,
                "reward_components/tool_reward_max":
                    max(tool_scores) if tool_scores else 0.0,
            },
            commit=False,
        )

    def cal_format_score(self, message: List[Dict[str, Any]]) -> float:
        """
        Preserve the original multi-turn format rule:
        every assistant turn must contain exactly one of <answer> or <crop>.
        """
        is_correct = True
        for line in message:
            content = _message_content(line)
            if line["role"] == "assistant":
                has_answer = extract_answer(content) is not None
                has_crop = extract_crop(content) is not None

                if not has_answer and not has_crop:
                    is_correct = False
                    break
                if has_answer and has_crop:
                    is_correct = False
                    break

                # For a final answer turn, additionally enforce option_X format.
                if has_answer:
                    if self.normalize_option(extract_answer(content)) is None:
                        is_correct = False
                        break

        return 1.0 if is_correct else 0.0

    def cal_tool_score(self, message: List[Dict[str, Any]]) -> float:
        """
        Preserve the original crop-tool reward behavior.
        """
        num_image = 0
        num_round = 0

        for line in message:
            if line["role"] == "user":
                num_image += _message_content(line).count(
                    "<ToolResponse><image></ToolResponse>"
                )

            if line["role"] == "assistant":
                num_round += 1
                content = _message_content(line)
                if extract_answer(content) is not None:
                    num_round -= 1

        return num_image / num_round if num_round > 0 else 0.0


# Keep the original registration names to avoid changing your training command.
orms["doc_crop_reward"] = DocCropReward
multi_turns["doc_crop_scheduler"] = DocCropScheduler
