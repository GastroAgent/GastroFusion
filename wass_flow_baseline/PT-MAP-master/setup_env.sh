#!/usr/bin/env bash
# Create the conda environment for PT-MAP (medical few-shot pretrain).
set -euo pipefail

ENV_NAME="${ENV_NAME:-fsl_ptmap}"
PY_VER="${PY_VER:-3.10}"
TORCH_VER="${TORCH_VER:-2.1.0}"
CUDA_TAG="${CUDA_TAG:-cu118}"

echo "==> conda env: ${ENV_NAME} (python ${PY_VER}, torch ${TORCH_VER}+${CUDA_TAG})"

conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install "torch==${TORCH_VER}+${CUDA_TAG}" \
                     "torchvision==0.16.0+${CUDA_TAG}" \
                     --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -m pip install "numpy<2" matplotlib tqdm Pillow

# sanity check
python -c "import torch, torchvision, numpy; \
    print('torch', torch.__version__, 'cuda', torch.version.cuda, \
          'avail', torch.cuda.is_available(), 'numpy', numpy.__version__)"

echo "==> done. activate with:  conda activate ${ENV_NAME}"