"""
GastroMLLM 推理服务 (Qwen3-VL-30B-A3B-Instruct-FP8)
基于 FastAPI 提供 HTTP API 接口
接口与原LLaVA服务完全兼容，GastroAgent无需改动

用法: python gastro_mllm_qwen3vl_server.py --port 8001
"""
import os
import sys
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from contextlib import asynccontextmanager

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==================== 配置 ====================
MODEL_PATH = "/mnt/inaisfs/data/home/tansy_criait/weights/Qwen3-VL-30B-A3B-Instruct-FP8"
# CUDA_VISIBLE_DEVICES 只是“可见卡编号”环境变量，不是 torch.device 字符串。
# 模型使用 device_map="auto"；输入放到当前进程可见的默认 CUDA 设备即可。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DATABASE_PATH = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy/胃_doctor"

# ==================== 全局模型 ====================
model = None
processor = None

# ==================== API 模型 ====================
class VisionAnalyzeRequest(BaseModel):
    images: List[str] = Field(..., description="图像路径列表")
    prompt: str = Field(..., description="诊断问题或指令")
    max_new_tokens: int = Field(512, description="最大生成token数")
    temperature: float = Field(0.6, description="采样温度")
    top_p: float = Field(0.95, description="核采样")
    do_sample: bool = Field(True, description="是否使用采样")
    repetition_penalty: float = Field(1.15, description="重复惩罚")

class VisionAnalyzeResponse(BaseModel):
    text: str = Field(..., description="模型生成的最终回答")
    thinking: Optional[str] = Field(None, description="思考过程")
    images: List[str] = Field(..., description="图像路径列表")


class VisionAnalyzeWithLogitsRequest(BaseModel):
    images: List[str] = Field(..., description="图像路径列表")
    prompt: str = Field(..., description="诊断问题或指令")
    max_new_tokens: int = Field(512, description="最大生成token数")
    temperature: float = Field(0.0, description="采样温度（logits模式固定为0）")
    top_p: float = Field(0.95, description="核采样")
    do_sample: bool = Field(False, description="是否使用采样（logits模式固定为False）")
    repetition_penalty: float = Field(1.15, description="重复惩罚")
    candidate_classes: List[str] = Field(
        default_factory=list,
        description="候选疾病名称列表，用于计算各类别的 logits 置信度"
    )


class DiseaseConfidence(BaseModel):
    disease: str = Field(..., description="疾病名称")
    logit: float = Field(..., description="该疾病第一个 token 的原始 logit 值")
    mean_logit: float = Field(..., description="该疾病所有 token 的平均 logit")
    token_count: int = Field(..., description="该疾病名称的 token 数量")
    confidence: float = Field(..., description="归一化置信度 (0-1)")


class VisionAnalyzeWithLogitsResponse(BaseModel):
    text: str = Field(..., description="模型生成的最终回答")
    thinking: Optional[str] = Field(None, description="思考过程")
    images: List[str] = Field(..., description="图像路径列表")
    disease_confidences: List[DiseaseConfidence] = Field(
        default_factory=list,
        description="候选疾病名称的 logits 置信度列表，按 confidence 降序"
    )
    raw_text: str = Field(..., description="原始输出文本（未经思考过程过滤）")


# ==================== Teacher-forcing 候选打分 ====================
class ScoreCandidatesRequest(BaseModel):
    images: List[str] = Field(..., description="图像路径列表")
    candidate_classes: List[str] = Field(..., description="候选疾病名称列表")
    class_profiles: Optional[dict] = Field(
        default=None,
        description="可选：{疾病名: 视觉特征描述} 字典，注入到 prompt 中辅助判别"
    )
    softmax_temperature: float = Field(
        1.0, gt=0, description="跨候选归一化时的 softmax 温度；越大越平滑"
    )
    length_normalize: bool = Field(
        True,
        description="True 时跨候选用 mean_log_prob 做 softmax（消除长度偏置）；False 用 joint log_prob"
    )


class CandidateScore(BaseModel):
    disease: str = Field(..., description="疾病名称")
    log_prob: float = Field(..., description="log P(disease | image, prompt) — 联合 log-prob")
    mean_log_prob: float = Field(..., description="按 token 数归一的 mean log-prob（去掉长度偏置）")
    norm_prob: float = Field(..., description="跨候选 softmax 归一化后的概率 (0-1)")
    token_count: int = Field(..., description="该疾病名 token 数")


class ScoreCandidatesResponse(BaseModel):
    images: List[str] = Field(..., description="图像路径列表")
    candidates: List[CandidateScore] = Field(
        ..., description="按 norm_prob 降序排列的候选打分列表"
    )


# ==================== MLLM -> FSL 候选疾病接口 ====================
class SupportClassItem(BaseModel):
    """
    FSL 的候选类别元信息。
    label_name 是疾病标准名；support_dir 可直接交给 FSL /calculate_wasserstein。
    visual_profile / aliases 都是可选的。
    """
    label_name: str = Field(..., description="疾病标准名称")
    support_dir: Optional[str] = Field(None, description="该疾病对应的 FSL support 目录")
    visual_profile: Optional[str] = Field(None, description="可选：该类别的视觉画像")
    aliases: List[str] = Field(default_factory=list, description="可选：疾病别名")


class CandidateDiseasesRequest(BaseModel):
    """
    输入一张 query 图像 + 当前任务所有可用 support 类别，
    由 MLLM 输出可直接用于 FSL 子集分类的 candidate diseases。
    """
    target_image: str = Field(..., description="query 图像路径")
    support_classes: List[SupportClassItem] = Field(
        ..., description="当前 FSL 可用疾病类别及 support_dir"
    )
    top_k: int = Field(5, ge=1, description="输出给 FSL 的候选疾病数")
    max_new_tokens: int = Field(384, ge=32, description="开放式候选生成最大 token 数")

    # 默认保留 teacher-forcing 作为候选补全/稳定性打分。
    # 它不会直接执行 FSL，只返回候选疾病及 support_dirs。
    use_teacher_forcing: bool = Field(
        True,
        description="是否对已知疾病谱做 teacher-forcing 打分，用于补全候选及输出置信度"
    )
    softmax_temperature: float = Field(
        1.0, gt=0, description="teacher-forcing 候选归一化温度"
    )
    length_normalize: bool = Field(
        True, description="teacher-forcing 是否使用 mean log-prob 消除名称长度偏置"
    )


class CandidateDiseaseItem(BaseModel):
    rank: int
    disease: str
    support_dir: Optional[str] = None
    source: str = Field(
        ..., description="open_generation / teacher_force_fill / fallback"
    )
    norm_prob: Optional[float] = None
    log_prob: Optional[float] = None
    mean_log_prob: Optional[float] = None


class CandidateDiseasesResponse(BaseModel):
    ok: bool = True
    target_image: str
    diag: str

    # MLLM open-generation Top-k.
    # NOTE: these candidates are NOT used to prune/restrict FSL in the current
    # offline full-set fusion protocol. They are kept as MLLM diagnostic output.
    candidate_diseases: List[str]
    candidate_support_dirs: List[str]
    candidates: List[CandidateDiseaseItem]

    # IMPORTANT:
    # Full disease-set teacher-forcing scores for V2 fusion.
    #
    # Expected downstream format:
    # {
    #   "class_a": {
    #       "norm_prob": ...,
    #       "log_prob": ...,
    #       "mean_log_prob": ...,
    #       "token_count": ...
    #   },
    #   ...
    # }
    #
    # The V2 fusion script reads mllm_score_map[class_name]["norm_prob"].
    # Therefore this map should contain ALL support_classes, not only Top-k.
    mllm_score_map: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    visual_evidence: str = ""
    risk_level: str = ""
    raw_text: str = ""


# ==================== Candidate diseases 辅助函数 ====================
def _extract_json_object(text: str) -> Dict[str, Any]:
    """尽量从 MLLM 文本中解析 JSON；失败则保留原文本。"""
    if not isinstance(text, str):
        return {}

    raw = text.strip()
    if not raw:
        return {}

    try:
        obj = __import__("json").loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw, flags=re.S)
        if not m:
            continue
        candidate = m.group(1)
        try:
            obj = __import__("json").loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {
        "diag": raw[:120],
        "topk": [],
        "visual_evidence": raw,
        "risk_level": "",
    }


def _norm_label_text(s: str) -> str:
    """仅用于把自由生成名称映射回当前 support_classes，不改变标准 label 本身。"""
    s = str(s or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-—–·,:：;；/\\\\]+", "", s)
    return s


def _match_to_known_label(
    raw_label: Any,
    support_classes: List[SupportClassItem],
) -> Optional[str]:
    """
    将 MLLM 自由生成疾病名映射到 support_classes 中的标准名。
    顺序：
      1) 标准名严格/归一化匹配
      2) aliases 匹配
      3) 安全的包含关系匹配
    """
    raw = str(raw_label or "").strip()
    if not raw:
        return None

    raw_norm = _norm_label_text(raw)

    # exact normalized canonical
    for item in support_classes:
        if raw == item.label_name or raw_norm == _norm_label_text(item.label_name):
            return item.label_name

    # aliases
    for item in support_classes:
        for alias in item.aliases:
            if raw_norm == _norm_label_text(alias):
                return item.label_name

    # containment fallback; choose the longest matching known string to reduce ambiguity
    matches = []
    for item in support_classes:
        can_norm = _norm_label_text(item.label_name)
        if can_norm and (can_norm in raw_norm or raw_norm in can_norm):
            matches.append((len(can_norm), item.label_name))
        for alias in item.aliases:
            alias_norm = _norm_label_text(alias)
            if alias_norm and (alias_norm in raw_norm or raw_norm in alias_norm):
                matches.append((len(alias_norm), item.label_name))

    if matches:
        matches.sort(reverse=True)
        return matches[0][1]

    return None


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_candidate_prompt(
    support_classes: List[SupportClassItem],
    top_k: int,
) -> str:
    disease_lines = []
    for item in support_classes:
        if item.visual_profile:
            disease_lines.append(
                f"- {item.label_name}：{item.visual_profile}"
            )
        else:
            disease_lines.append(f"- {item.label_name}")

    disease_text = "\n".join(disease_lines)

    return f"""你是消化内镜图像诊断助手。

本系统当前 FSL support set 对应的已知疾病谱系如下：
{disease_text}

请根据当前内镜图像进行开放式诊断，并生成最可能的候选疾病。
这些候选疾病将直接作为后续 few-shot learning (FSL) 的候选类别子集。

要求：
1. 不使用 A/B/C/D 或选择题编号；
2. 必须输出完整疾病名称；
3. 疾病名称应尽量严格使用上方疾病谱系中的标准名称；
4. 输出最多 {top_k} 个候选，按可能性从高到低排列；
5. diag 必须是你认为最可能的一个疾病；
6. 只输出 JSON，不要输出 JSON 之外的解释。

输出格式：
{{
  "diag": "最可能诊断名称",
  "topk": ["候选疾病1", "候选疾病2", "候选疾病3"],
  "visual_evidence": "当前图像中支持该判断的关键视觉依据",
  "risk_level": "low/medium/high"
}}"""


async def _teacher_force_all_known_classes(
    target_image: str,
    support_classes: List[SupportClassItem],
    softmax_temperature: float,
    length_normalize: bool,
) -> Dict[str, CandidateScore]:
    """
    复用本服务已有 /score_candidates 的核心 endpoint。
    当前器官通常只有 7/8 个类别，因此对整个已知疾病谱打分成本可控。
    """
    class_profiles = {
        x.label_name: x.visual_profile
        for x in support_classes
        if x.visual_profile
    }

    req = ScoreCandidatesRequest(
        images=[target_image],
        candidate_classes=[x.label_name for x in support_classes],
        class_profiles=class_profiles,
        softmax_temperature=softmax_temperature,
        length_normalize=length_normalize,
    )
    response = await vision_score_candidates(req)
    return {x.disease: x for x in response.candidates}


def _serialize_full_score_map(
    score_map: Dict[str, CandidateScore],
    support_classes: List[SupportClassItem],
) -> Dict[str, Dict[str, float]]:
    """
    Convert teacher-forcing CandidateScore objects to the exact JSON dict
    consumed by the offline V2 fusion code.

    The returned map is FULL-SET: every support class is present.
    If a class somehow has no score, a conservative zero-probability entry
    is emitted instead of dropping the key.
    """
    out: Dict[str, Dict[str, float]] = {}

    for item in support_classes:
        disease = item.label_name
        score = score_map.get(disease)

        if score is None:
            out[disease] = {
                "log_prob": -1.0e9,
                "mean_log_prob": -1.0e9,
                "norm_prob": 0.0,
                "token_count": 0,
            }
            continue

        out[disease] = {
            "log_prob": float(score.log_prob),
            "mean_log_prob": float(score.mean_log_prob),
            "norm_prob": float(score.norm_prob),
            "token_count": int(score.token_count),
        }

    return out


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="GastroMLLM 推理服务 (Qwen3-VL-30B-A3B-FP8)",
    description="消化内镜视觉大模型推理 API；支持直接输出 FSL candidate diseases",
    version="4.1.0",
    lifespan=lifespan
)

def load_model():
    global model, processor
    print(f"🔄 正在加载模型: {MODEL_PATH}")

    try:
        # 使用 AutoModelForImageTextToText 匹配 Qwen3VLMoeForConditionalGeneration 架构
        # torch_dtype="auto" 让 transformers 自动处理 FP8 权重
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("✅ 模型加载成功！")
        print(f"   架构: {model.__class__.__name__}")
        print(f"   设备分布: {model.hf_device_map if hasattr(model, 'hf_device_map') else DEVICE}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

def load_image(image_id: str) -> str:
    """
    解析图像路径，返回可用的路径字符串
    Qwen-VL 的 process_vision_info 可以直接处理本地路径
    """
    image_id = str(image_id).strip()

    # 解析 [Image: source:xxx] 格式
    if image_id.startswith("[Image:") and "source:" in image_id:
        match = re.search(r'source:\s*([^\]]+)', image_id)
        if match:
            image_id = match.group(1).strip()

    # 本地绝对路径
    if os.path.exists(image_id):
        return image_id

    # 在图像数据库中查找
    possible_paths = [
        os.path.join(IMAGE_DATABASE_PATH, f"{image_id}.jpg"),
        os.path.join(IMAGE_DATABASE_PATH, f"{image_id}.png"),
        os.path.join(IMAGE_DATABASE_PATH, image_id),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path

    # URL 直接返回
    if image_id.startswith("http://") or image_id.startswith("https://"):
        return image_id

    raise FileNotFoundError(f"无法找到图像: {image_id}")

@app.get("/")
async def root():
    return {
        "service": "GastroMLLM (Qwen3-VL-30B-A3B-FP8)",
        "status": "running",
        "model": MODEL_PATH,
        "device": DEVICE,
        "architecture": "Qwen3VLMoeForConditionalGeneration"
    }

@app.post("/v1/vision/analyze", response_model=VisionAnalyzeResponse)
async def vision_analyze(request: VisionAnalyzeRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 1. 解析图像路径
        image_paths = []
        for img_path in request.images:
            try:
                resolved = load_image(img_path)
                image_paths.append(resolved)
            except Exception as e:
                print(f"⚠️ 图像加载失败 {img_path}: {e}")
                continue

        if not image_paths and len(request.images) > 0:
            raise HTTPException(status_code=400, detail="所有图像加载均失败")

        # 2. 构建 Qwen-VL messages 格式
        sys_prompt = (
            "You are a helpful Medical AI assistant and an authoritative expert "
            "in the field of medicine. You need to understand the visual content "
            "and use natural language to assist users in various tasks."
        )

        # 构建 user content: 图像 + 文本
        user_content = []
        for img_path in image_paths:
            user_content.append({"type": "image", "image": f"file://{img_path}","max_pixels": 501760})
        user_content.append({"type": "text", "text": request.prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
        ]

        # 3. 处理输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)

        # 4. 推理
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.do_sample else None,
                top_p=request.top_p if request.do_sample else None,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                use_cache=True,
            )

        # 5. 解码（只保留新生成的部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # 6. 解析 <think>（如果模型输出了思考过程）
        thinking = None
        result = output_text

        if "<think>" in result and "</think>" in result:
            thinking_start = result.index("<think>") + len("<think>")
            thinking_end = result.index("</think>")
            thinking = result[thinking_start:thinking_end].strip()

            remaining_text = result[thinking_end + len("</think>"):].strip()
            if "<answer>" in remaining_text and "</answer>" in remaining_text:
                ans_start = remaining_text.index("<answer>") + len("<answer>")
                ans_end = remaining_text.index("</answer>")
                result = remaining_text[ans_start:ans_end].strip()
            else:
                result = remaining_text

        return VisionAnalyzeResponse(
            text=result,
            thinking=thinking,
            images=request.images
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.post("/v1/vision/batch_analyze")
async def batch_analyze(requests_list: List[VisionAnalyzeRequest]):
    results = []
    for req in requests_list:
        try:
            result = await vision_analyze(req)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "image_id": req.images[0] if req.images else "unknown"
            })
    return results


@app.post("/v1/vision/analyze_with_logits", response_model=VisionAnalyzeWithLogitsResponse)
async def vision_analyze_with_logits(request: VisionAnalyzeWithLogitsRequest):
    """
    带 logits 置信度的开放式生成端点。

    核心逻辑：
    1. 用 greedy 模式（do_sample=False）生成回答，获取稳定的 logits；
    2. 解码生成文本；
    3. 将 candidate_classes 中的每个疾病名称 tokenize；
    4. 在 output logits 上，对每个疾病名称的 token 序列计算 logit sum / mean；
    5. 对所有候选疾病做归一化 softmax，得到每个疾病的置信度分数。
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # ---------- 1. 解析图像路径 ----------
        image_paths = []
        for img_path in request.images:
            try:
                resolved = load_image(img_path)
                image_paths.append(resolved)
            except Exception as e:
                print(f"⚠️ 图像加载失败 {img_path}: {e}")
                continue

        if not image_paths and len(request.images) > 0:
            raise HTTPException(status_code=400, detail="所有图像加载均失败")

        # ---------- 2. 构建 messages ----------
        sys_prompt = (
            "You are a helpful Medical AI assistant and an authoritative expert "
            "in the field of medicine. You need to understand the visual content "
            "and use natural language to assist users in various tasks."
        )
        user_content = []
        for img_path in image_paths:
            user_content.append({"type": "image", "image": f"file://{img_path}","max_pixels": 501760})
        user_content.append({"type": "text", "text": request.prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)
        input_len = inputs.input_ids.shape[1]

        # ---------- 3. 推理 + 捕获 logits ----------
        # do_sample=False 确保 logits 稳定且可解释
        with torch.inference_mode():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=None,
                top_p=None,
                do_sample=False,
                repetition_penalty=request.repetition_penalty,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
            generated_ids = generation_output.sequences
            scores = generation_output.scores  # tuple of (vocab_size,) tensors per step

        # ---------- 4. 解码生成文本 ----------
        gen_trimmed = generated_ids[0][input_len:]
        output_text = processor.decode(
            gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        raw_text = output_text

        thinking = None
        if "<think>" in output_text and "</think>" in output_text:
            ts = output_text.index("<think>") + len("<think>")
            te = output_text.index("</think>")
            thinking = output_text[ts:te].strip()
            remaining = output_text[te + len("</think>"):].strip()
            if "<answer>" in remaining and "</answer>" in remaining:
                as_ = remaining.index("<answer>") + len("<answer>")
                ae = remaining.index("</answer>")
                output_text = remaining[as_:ae].strip()
            else:
                output_text = remaining

        # ---------- 5. 计算每个候选疾病的 logits 置信度 ----------
        disease_confidences: List[DiseaseConfidence] = []
        if request.candidate_classes and scores and len(scores) > 0:
            # scores: tuple of [batch_size, vocab_size]，stack 后 [seq_len, batch=1, vocab_size]
            logits_seq = torch.stack(scores, dim=0)

            disease_logits_map = {}
            for disease in request.candidate_classes:
                disease_ids = processor.tokenizer(
                    disease, add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]
                n_tok = len(disease_ids)

                if n_tok == 0:
                    disease_logits_map[disease] = {"logit": 0.0, "mean_logit": 0.0, "n_tok": 0}
                    continue

                # 在每个生成步的 logits 分布上，取该疾病名对应 token 的 logit 值
                # 这才是"该候选疾病在当前位置被选中的可能性"
                n_steps = min(n_tok, logits_seq.shape[0])
                token_logits = [
                    logits_seq[i, 0, disease_ids[i]].item()
                    for i in range(n_steps)
                ]

                if token_logits:
                    disease_logits_map[disease] = {
                        "logit": token_logits[0],
                        "mean_logit": sum(token_logits) / len(token_logits),
                        "n_tok": n_tok,
                    }
                else:
                    disease_logits_map[disease] = {"logit": 0.0, "mean_logit": 0.0, "n_tok": n_tok}

            # ---------- 归一化：shifted softmax ----------
            valid = [
                (d, info["logit"], info["mean_logit"], info["n_tok"])
                for d, info in disease_logits_map.items()
                if info["n_tok"] > 0
            ]
            if valid:
                mean_logits = [v[2] for v in valid]
                max_ml = max(mean_logits)

                # shifted softmax 归一化：避免数值溢出
                exp_vals = [float(torch.exp(torch.tensor(m - max_ml))) for m in mean_logits]
                exp_sum = sum(exp_vals)

                for i, (disease, logit, mean_logit, n_tok) in enumerate(valid):
                    conf = exp_vals[i] / exp_sum if exp_sum > 0 else 0.0
                    disease_confidences.append(DiseaseConfidence(
                        disease=disease,
                        logit=round(logit, 4),
                        mean_logit=round(mean_logit, 4),
                        token_count=n_tok,
                        confidence=round(conf, 4),
                    ))

                disease_confidences.sort(key=lambda x: x.confidence, reverse=True)

        return VisionAnalyzeWithLogitsResponse(
            text=output_text,
            thinking=thinking,
            images=request.images,
            disease_confidences=disease_confidences,
            raw_text=raw_text,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"带 logits 推理失败: {str(e)}")


@app.post("/v1/vision/score_candidates", response_model=ScoreCandidatesResponse)
async def vision_score_candidates(request: ScoreCandidatesRequest):
    """
    Teacher-forcing 候选打分。

    流程：
    1. 构建 messages（图像 + 系统/用户 prompt，最后留 assistant 生成位）；
    2. 用 chat template 得到 prefix_text（生成位之前的全部内容）；
    3. 对每个候选疾病 D：
       - full_text = prefix_text + D
       - 单次前向取 logits
       - 取 prefix 之后那些位置的 log_softmax，对 D 的 token id 取出来求和
       - log_prob = log P(D | image, prompt)
    4. 跨候选做 shifted softmax 归一化（默认按 mean_log_prob 以去除长度偏置）；
    5. 按 norm_prob 降序返回。

    与 /v1/vision/analyze_with_logits 的区别：
    - 这里 logit 是"在生成位上预测 disease token 的真概率"，语义干净；
    - 不依赖模型自由生成的 JSON 结构；
    - 阈值跨数据集稳定。
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if not request.candidate_classes:
        raise HTTPException(status_code=400, detail="candidate_classes 不能为空")

    try:
        # ---------- 1. 解析图像 ----------
        image_paths = []
        for img_path in request.images:
            try:
                image_paths.append(load_image(img_path))
            except Exception as e:
                print(f"⚠️ 图像加载失败 {img_path}: {e}")
                continue
        if not image_paths and len(request.images) > 0:
            raise HTTPException(status_code=400, detail="所有图像加载均失败")

        # ---------- 2. 构建 messages（专为打分设计的极简 prompt） ----------
        sys_prompt = (
            "You are a helpful Medical AI assistant and an authoritative expert "
            "in the field of medicine. You need to understand the visual content "
            "and use natural language to assist users in various tasks."
        )

        # 候选疾病视觉画像（可选注入）
        profiles = request.class_profiles or {}
        disease_lines = []
        for c in request.candidate_classes:
            profile = profiles.get(c, "")
            if profile:
                disease_lines.append(f"- {c}：{profile}")
            else:
                disease_lines.append(f"- {c}")
        disease_text = "\n".join(disease_lines)

        user_text = (
            "你是消化内镜图像诊断助手。\n\n"
            "本系统的已知疾病谱系：\n"
            f"{disease_text}\n\n"
            "请直接给出最可能的疾病名称（只输出名称，不要任何其它内容）。\n\n"
            "最可能的诊断："
        )

        user_content = []
        for img_path in image_paths:
            user_content.append({"type": "image", "image": f"file://{img_path}","max_pixels": 501760})
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
        ]

        # ---------- 3. 处理 prefix（共享）----------
        prefix_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        prefix_inputs = processor(
            text=[prefix_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prefix_len = int(prefix_inputs.input_ids.shape[1])

        # ---------- 4. 对每个候选 teacher-force 打分 ----------
        candidate_scores = []   # list of (disease, log_prob, mean_log_prob, n_tok)
        for disease in request.candidate_classes:
            # 用 "\n" 分隔，避免 BPE 与前缀末尾字符合并
            full_text = prefix_text + disease

            full_inputs = processor(
                text=[full_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            full_ids = full_inputs.input_ids[0]
            full_len = int(full_ids.shape[0])
            n_tok = full_len - prefix_len

            if n_tok <= 0:
                # 边界情况：prefix 与 full 长度一致（疾病名被 tokenize 为 0 token，几乎不会发生）
                candidate_scores.append((disease, float("-inf"), float("-inf"), 0))
                continue

            with torch.inference_mode():
                outputs = model(**full_inputs, use_cache=False)
            # logits: [1, full_len, vocab]
            logits = outputs.logits

            # 位置 i 上的 logits 用于预测位置 i+1 的 token
            # 我们要预测的 disease token 位于 [prefix_len, full_len)
            # 因此用 logits[0, prefix_len-1 : full_len-1, :]
            target_logits = logits[0, prefix_len - 1 : full_len - 1, :].float()
            log_probs = torch.log_softmax(target_logits, dim=-1)
            target_ids = full_ids[prefix_len:full_len].to(log_probs.device)
            token_log_probs = log_probs[torch.arange(n_tok), target_ids]
            joint_lp = float(token_log_probs.sum().item())
            mean_lp = joint_lp / n_tok
            candidate_scores.append((disease, joint_lp, mean_lp, n_tok))

        # ---------- 5. 跨候选 softmax 归一化 ----------
        if request.length_normalize:
            base_scores = [s[2] for s in candidate_scores]  # mean_log_prob
        else:
            base_scores = [s[1] for s in candidate_scores]

        # shifted softmax，避免 inf/溢出
        T = request.softmax_temperature
        finite = [v for v in base_scores if v != float("-inf")]
        if finite:
            max_v = max(finite)
            exps = []
            for v in base_scores:
                if v == float("-inf"):
                    exps.append(0.0)
                else:
                    exps.append(float(torch.exp(torch.tensor((v - max_v) / T))))
            exp_sum = sum(exps) or 1.0
            norm_probs = [e / exp_sum for e in exps]
        else:
            norm_probs = [0.0] * len(base_scores)

        results = []
        for (disease, lp, mlp, n_tok), p in zip(candidate_scores, norm_probs):
            results.append(CandidateScore(
                disease=disease,
                log_prob=round(lp, 6) if lp != float("-inf") else -1e9,
                mean_log_prob=round(mlp, 6) if mlp != float("-inf") else -1e9,
                norm_prob=round(p, 6),
                token_count=n_tok,
            ))
        results.sort(key=lambda r: r.norm_prob, reverse=True)

        return ScoreCandidatesResponse(images=request.images, candidates=results)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"打分失败: {str(e)}")


# ==================== MLLM candidate diseases -> FSL ====================
@app.post(
    "/v1/vision/candidate_diseases",
    response_model=CandidateDiseasesResponse,
)
async def candidate_diseases(request: CandidateDiseasesRequest):
    """
    MLLM 候选疾病服务。

    输入:
        target_image
        support_classes = [{label_name, support_dir, ...}, ...]

    输出:
        candidate_diseases      -> 疾病标准名列表
        candidate_support_dirs  -> 与上述疾病一一对应，可直接传给 FSL
        candidates              -> 带 rank/source/teacher-forcing score 的详细信息

    本 endpoint 本身不执行 FSL。

    当前 full-set offline fusion 协议：
        1) 本接口输出 MLLM candidate_diseases（诊断信息）；
        2) 本接口同时输出 FULL-SET mllm_score_map；
        3) FSL 不使用 candidate_diseases 做裁剪；
        4) FSL ranking/top1/gap 从 offline full-set RAE 结果读取；
        5) V2 fusion 使用：
               offline full-set FSL view
               +
               本接口的 full-set mllm_score_map
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if not request.support_classes:
        raise HTTPException(status_code=400, detail="support_classes 不能为空")

    # 去掉重复 label，保持输入顺序。
    unique_support = []
    seen_labels = set()
    for item in request.support_classes:
        name = item.label_name.strip()
        if not name or name in seen_labels:
            continue
        seen_labels.add(name)
        unique_support.append(item)

    if not unique_support:
        raise HTTPException(status_code=400, detail="没有有效的 support class")

    top_k = min(int(request.top_k), len(unique_support))

    try:
        # ------------------------------------------------------------
        # Step 1. 开放式 MLLM 生成 Top-k
        # ------------------------------------------------------------
        prompt = _build_candidate_prompt(unique_support, top_k)

        open_req = VisionAnalyzeRequest(
            images=[request.target_image],
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05,
        )
        open_resp = await vision_analyze(open_req)
        raw_text = open_resp.text or ""
        parsed = _extract_json_object(raw_text)

        raw_topk = []
        for key in ("topk", "candidate_diseases", "candidate_classes", "candidates", "top_k"):
            value = parsed.get(key)
            if isinstance(value, list):
                raw_topk.extend(value)

        raw_diag = (
            parsed.get("diag")
            or parsed.get("diagnosis")
            or parsed.get("pred")
            or ""
        )
        if raw_diag:
            # diag 放在最前，但去重后不会重复占位。
            raw_topk = [raw_diag] + raw_topk

        open_candidates = []
        for raw_name in raw_topk:
            matched = _match_to_known_label(raw_name, unique_support)
            if matched:
                open_candidates.append(matched)
        open_candidates = _dedupe_keep_order(open_candidates)

        # ------------------------------------------------------------
        # Step 2. 对 FULL disease set 做 teacher-forcing 打分
        #
        # 这里的 score_map 与 candidate_diseases 是两件不同的事：
        #
        #   candidate_diseases:
        #       MLLM 自由生成 Top-k，仅作为 MLLM 诊断输出。
        #
        #   mllm_score_map:
        #       对 support_classes 中所有类别做 teacher-forcing，
        #       作为后续 V2 fusion 的 MLLM 概率输入。
        #
        # 当前协议下，candidate_diseases 不限制 FSL。
        # FSL 仍从 offline full-set RAE 结果读取。
        # ------------------------------------------------------------
        score_map: Dict[str, CandidateScore] = {}
        if request.use_teacher_forcing:
            score_map = await _teacher_force_all_known_classes(
                target_image=request.target_image,
                support_classes=unique_support,
                softmax_temperature=request.softmax_temperature,
                length_normalize=request.length_normalize,
            )

        # ------------------------------------------------------------
        # Step 3. 最终候选
        #   A. 优先保留开放生成并成功映射到 support classes 的疾病
        #   B. 不足 top_k 时按 teacher-forcing norm_prob 补全
        #   C. teacher forcing 关闭/失败时按 support_classes 固定顺序兜底
        # ------------------------------------------------------------
        final_names = list(open_candidates[:top_k])
        source_map = {x: "open_generation" for x in final_names}

        if len(final_names) < top_k and score_map:
            scored_order = sorted(
                score_map.values(),
                key=lambda x: x.norm_prob,
                reverse=True,
            )
            for item in scored_order:
                if item.disease not in final_names:
                    final_names.append(item.disease)
                    source_map[item.disease] = "teacher_force_fill"
                if len(final_names) >= top_k:
                    break

        if len(final_names) < top_k:
            for item in unique_support:
                if item.label_name not in final_names:
                    final_names.append(item.label_name)
                    source_map[item.label_name] = "fallback"
                if len(final_names) >= top_k:
                    break

        support_map = {
            item.label_name: item.support_dir
            for item in unique_support
        }

        detailed = []
        for rank, disease in enumerate(final_names, start=1):
            score = score_map.get(disease)
            detailed.append(
                CandidateDiseaseItem(
                    rank=rank,
                    disease=disease,
                    support_dir=support_map.get(disease),
                    source=source_map.get(disease, "fallback"),
                    norm_prob=(
                        float(score.norm_prob)
                        if score is not None
                        else None
                    ),
                    log_prob=(
                        float(score.log_prob)
                        if score is not None
                        else None
                    ),
                    mean_log_prob=(
                        float(score.mean_log_prob)
                        if score is not None
                        else None
                    ),
                )
            )

        candidate_support_dirs = [
            support_map[d]
            for d in final_names
            if support_map.get(d)
        ]

        diag = _match_to_known_label(raw_diag, unique_support)
        if not diag:
            diag = final_names[0] if final_names else "未知"

        # FULL-SET score map for offline V2 fusion.
        # This is intentionally independent of final_names / Top-k.
        full_mllm_score_map = _serialize_full_score_map(
            score_map=score_map,
            support_classes=unique_support,
        )

        return CandidateDiseasesResponse(
            ok=True,
            target_image=request.target_image,
            diag=diag,
            candidate_diseases=final_names,
            candidate_support_dirs=candidate_support_dirs,
            candidates=detailed,
            mllm_score_map=full_mllm_score_map,
            visual_evidence=str(parsed.get("visual_evidence", "") or ""),
            risk_level=str(parsed.get("risk_level", "") or ""),
            raw_text=raw_text,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"candidate diseases 生成失败: {str(e)}",
        )


@app.post("/v1/vision/candidate_diseases/batch")
async def candidate_diseases_batch(
    requests_list: List[CandidateDiseasesRequest],
):
    """批量生成 FSL candidate diseases；逐条返回，单条失败不影响其它样本。"""
    outputs = []
    for req in requests_list:
        try:
            outputs.append(await candidate_diseases(req))
        except Exception as e:
            outputs.append({
                "ok": False,
                "target_image": req.target_image,
                "error": str(e),
            })
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    MODEL_PATH = args.model_path

    print("=" * 80)
    print("🏥 GastroMLLM Candidate Service (Qwen3-VL-30B-A3B-Instruct-FP8)")
    print("=" * 80)
    print(f"模型路径: {MODEL_PATH}")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"架构:     Qwen3VLMoeForConditionalGeneration (MoE)")
    print("候选接口: /v1/vision/candidate_diseases")
    print("=" * 80)

    uvicorn.run(app, host=args.host, port=args.port)
