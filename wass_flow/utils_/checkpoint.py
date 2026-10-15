from __future__ import annotations

import os
from typing import Any

import torch


def save_checkpoint(path: str, **state: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = path + ".tmp"
    torch.save(state, temp)
    os.replace(temp, path)


def load_metric_checkpoint(path: str, model, strict: bool = True) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("metric_encoder", checkpoint.get("model", checkpoint))
    model.load_state_dict(state, strict=strict)
    return checkpoint
