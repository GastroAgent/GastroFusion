#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_lora.py
=============

将 ms-swift 训练得到的 LoRA 权重融合（merge）回 Qwen3-VL-8B-Instruct 基座模型，
输出与原始模型完全同结构的权重，可直接用于后续 RLHF / 推理。

用法
----
    python merge_lora.py \\
        --model      /path/to/Qwen3-VL-8B-Instruct \\
        --lora_ckpt  /path/to/output/xxx/checkpoint-1128 \\
        --output_dir /path/to/output/xxx-merged \\
        --dtype      bfloat16          # 可选: bfloat16 / float16 / float32

依赖
----
    transformers / peft / torch / accelerate
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into Qwen3-VL base model (ms-swift style)."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to base model, e.g. .../Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--lora_ckpt", type=str, required=True,
        help="Path to LoRA checkpoint dir, e.g. .../checkpoint-1128",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where the merged model will be saved",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype of merged weights (default: bfloat16, same as training)",
    )
    parser.add_argument(
        "--max_shard_size", type=str, default="10GB",
        help="Max shard size for saved safetensors (default: 10GB)",
    )
    return parser.parse_args()


def check_paths(args: argparse.Namespace) -> None:
    """校验输入路径，并准备输出目录。"""
    errs = []
    for p, name in ((args.model, "--model"), (args.lora_ckpt, "--lora_ckpt")):
        if not os.path.isdir(p):
            errs.append(f"{name} 路径不存在或不是目录: {p}")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        errs.append(
            f"--output_dir 已存在且非空，为避免覆盖已有权重请换一个目录: {args.output_dir}"
        )

    if errs:
        for e in errs:
            print(f"[ERROR] {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)


def load_base_model(model_path: str, torch_dtype):
    """
    加载 Qwen3-VL 基座模型。

    依次尝试:
      1. Qwen3VLForConditionalGeneration
      2. AutoModelForImageTextToText
      3. AutoModelForCausalLM

    若默认 attention 实现（可能为 flash_attn）加载失败，
    会自动降级尝试 sdpa / eager，保证在没有 flash-attn 的环境也能融合。
    """
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
        model_type = "Qwen3VLForConditionalGeneration"
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as ModelCls
            model_type = "AutoModelForImageTextToText"
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelCls
            model_type = "AutoModelForCausalLM"

    attn_candidates = [None, "sdpa", "eager"]
    last_err = None
    for attn in attn_candidates:
        kwargs = dict(
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if attn is not None:
            kwargs["attn_implementation"] = attn
        try:
            print(f"      使用 {model_type} 加载, attn_implementation={attn}")
            model = ModelCls.from_pretrained(model_path, **kwargs)
            print("      基座模型加载成功。")
            return model, model_type
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"      [WARN] 加载失败 (attn={attn}): {e}")

    raise RuntimeError(
        f"基座模型加载失败，所有 attention 实现均不可用。最后错误: {last_err}"
    )


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("LoRA Merge Script")
    print("=" * 70)
    print(f"Base Model : {args.model}")
    print(f"LoRA CKPT  : {args.lora_ckpt}")
    print(f"Output Dir : {args.output_dir}")
    print(f"Dtype      : {args.dtype}")
    print("-" * 70)

    # 1. 路径校验
    check_paths(args)

    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # 2. 加载基座模型
    print("\n[1/5] 加载基座模型 ...")
    t0 = time.time()
    model, model_type = load_base_model(args.model, torch_dtype)
    print(f"      耗时 {time.time() - t0:.1f}s")

    # 3. 加载 LoRA 并融合
    print("\n[2/5] 加载 LoRA adapter ...")
    from peft import PeftModel

    t0 = time.time()
    peft_model = PeftModel.from_pretrained(
        model,
        args.lora_ckpt,
        is_trainable=False,
    )
    print(f"      耗时 {time.time() - t0:.1f}s")
    print(f"      LoRA modules: {len(peft_model.peft_config)} config(s)")

    print("\n[3/5] 执行 merge_and_unload() ...")
    t0 = time.time()
    merged_model = peft_model.merge_and_unload(progressbar=True)
    print(f"      耗时 {time.time() - t0:.1f}s")

    # 4. 保存融合后的权重（先移到 CPU，避免显存峰值）
    print("\n[4/5] 保存融合权重 ...")
    t0 = time.time()

    try:
        merged_model.to("cpu")
    except Exception as e:  # noqa: BLE001
        print(f"      [WARN] 移动到 CPU 失败（继续在 GPU 上保存）: {e}")

    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    # 保存 processor / tokenizer
    from transformers import AutoProcessor

    try:
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(args.model)
        processor.save_pretrained(args.output_dir)
        print(f"      processor 保存完成（耗时 {time.time() - t0:.1f}s）")
    except Exception as e:  # noqa: BLE001
        print(f"      [WARN] processor 保存失败，请手动检查: {e}")

    print(f"      权重保存耗时 {time.time() - t0:.1f}s")

    # 记录融合信息（备查）
    import torch as _torch
    import transformers as _tf
    import peft as _peft

    merge_info = {
        "base_model": args.model,
        "lora_ckpt": args.lora_ckpt,
        "output_dir": args.output_dir,
        "dtype": args.dtype,
        "model_class": model_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "transformers_version": _tf.__version__,
        "peft_version": _peft.__version__,
        "torch_version": _torch.__version__,
        "merge_method": "peft.merge_and_unload",
    }
    info_path = os.path.join(args.output_dir, "merge_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(merge_info, f, ensure_ascii=False, indent=2)
    print(f"      融合信息已保存: {info_path}")

    # 5. 结果校验
    print("\n[5/5] 校验输出文件 ...")
    cfg_path = os.path.join(args.output_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"校验失败: {cfg_path} 不存在")

    weight_files = sorted(
        glob.glob(os.path.join(args.output_dir, "*.safetensors"))
    ) or sorted(glob.glob(os.path.join(args.output_dir, "*.bin")))

    if not weight_files:
        raise RuntimeError(f"校验失败: {args.output_dir} 下未找到权重文件")
    if os.path.isfile(info_path):
        weight_files.append(info_path)

    print("-" * 70)
    for wf in weight_files:
        size_mb = os.path.getsize(wf) / (1024 * 1024)
        print(f"  {os.path.basename(wf):<40s} {size_mb:>10.1f} MB")
    print("-" * 70)
    print("\n✅ 融合完成！输出目录:")
    print(f"   {args.output_dir}")
    print("\n该目录结构与原模型一致，可直接作为 --model 用于 swift rlhf / 推理。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] 融合过程出现异常:")
        traceback.print_exc()
        sys.exit(1)