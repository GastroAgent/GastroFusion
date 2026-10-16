#!/usr/bin/env bash
#SBATCH -J demo_q2l_stomach
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# SLURM defaults to slurm-%j.out unless output paths are configured by the submitter.

set -euo pipefail

# Activate the appropriate CUDA/conda environment BEFORE launching this script.
# Example: conda activate flow
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${CODE_DIR}")"
PY="${CODE_DIR}/eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.py"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/demo_data/stomach}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${REPO_ROOT}/demo_weights/wass-flow-stomach}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results/q2l_stomach_testtest}"
Q2L_CHECKPOINT="${Q2L_CHECKPOINT:-${WEIGHT_ROOT}/stomach-test-test/胃-test-test/metric_q2l_m0b_rae_manifest.pt}"
DIT_CHECKPOINT="${DIT_CHECKPOINT:-${WEIGHT_ROOT}/otcfm_dit_L_step_265000.pt}"

for file in "${PY}" "${Q2L_CHECKPOINT}" "${DIT_CHECKPOINT}"; do
    [[ -f "${file}" ]] || { echo "[ERROR] Missing file: ${file}" >&2; exit 2; }
done
[[ -d "${DATA_ROOT}" ]] || { echo "[ERROR] Missing class pool: ${DATA_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

# The unmodified evaluator expects DATA_BASE/indomain_support_76/hospital/organ/class/image.
# Adapt the public DATA_ROOT/class/image layout without moving or duplicating images.
TEMP_DATA="$(mktemp -d)"
trap 'rm -rf "${TEMP_DATA}"' EXIT
mkdir -p "${TEMP_DATA}/indomain_support_76/华西"
ln -s "$(cd "${DATA_ROOT}" && pwd)" "${TEMP_DATA}/indomain_support_76/华西/胃"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[Demo Q2L / Q2S-Retained] data=${DATA_ROOT}"
echo "[Demo Q2L / Q2S-Retained] q2l=${Q2L_CHECKPOINT}"
echo "[Demo Q2L / Q2S-Retained] dit=${DIT_CHECKPOINT}"
echo "[Demo Q2L / Q2S-Retained] output=${OUTPUT_DIR}"

# Import the existing Python file without triggering its __main__ block; override
# only USER CONFIG globals in memory. No model or evaluation code is edited.
python - "${PY}" "${CODE_DIR}" "${TEMP_DATA}" "${Q2L_CHECKPOINT}" "${DIT_CHECKPOINT}" "${OUTPUT_DIR}" <<'PY'
import importlib.util
import os
import sys

py_file, project_root, data_base, q2l_checkpoint, dit_checkpoint, out_dir = sys.argv[1:]
sys.path.insert(0, project_root)
spec = importlib.util.spec_from_file_location("gastro_q2l_stomach_demo", py_file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

module.PROJECT_ROOT = project_root
module.DATA_BASE = data_base
module.EVAL_SETTING = "test_test"
module.HOSPITALS = ["华西"]
module.PARTS = ["胃"]
module.N_WAY = 7  # Effective way is min(7, count of classes with >=6 images).
module.Q2L_CHECKPOINTS = {"胃-test-test": q2l_checkpoint}
module.DIT_CHECKPOINT_OVERRIDE = dit_checkpoint
module.BASE_RESULT_DIR = out_dir
module.RESULT_DIR = os.path.join(out_dir, module.get_fusion_tag())
# Keep the original evaluator's 5-shot / 1-query / 100 episodes / seed=42,
# soft-min, score-fusion rules, and WITHOUT-replacement test-test sampling.
module.main()
PY
