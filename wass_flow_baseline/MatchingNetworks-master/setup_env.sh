#!/usr/bin/env bash
# Create the conda environment for MatchingNetworks (medical few-shot).
# Run once on the GPU host before launching training.
set -euo pipefail

ENV_NAME="${ENV_NAME:-fsl_matching}"
PY_VER="${PY_VER:-3.10}"
TORCH_VER="${TORCH_VER:-2.1.0}"
CUDA_TAG="${CUDA_TAG:-cu118}"   # GPU33 uses CUDA 11.8

echo "==> conda env: ${ENV_NAME} (python ${PY_VER}, torch ${TORCH_VER}+${CUDA_TAG})"

conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Match your fsl_om env approach: install PyTorch via pip for the right CUDA build.
python -m pip install --upgrade pip
python -m pip install "torch==${TORCH_VER}+${CUDA_TAG}" \
                     "torchvision==0.16.0+${CUDA_TAG}" \
                     --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -m pip install "numpy<2" tqdm Pillow tensorboard_logger

# sanity check
python -c "import torch, torchvision; \
    print('torch', torch.__version__, 'cuda', torch.version.cuda, \
          'avail', torch.cuda.is_available()); \
    import tensorboard_logger; print('tensorboard_logger ok')"

echo "==> done. activate with:  conda activate ${ENV_NAME}"