#!/bin/bash
#SBATCH -J eval_f1_v5_pretrain_testtrain_reduce
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.err

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# export CUDA_VISIBLE_DEVICES=2

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_testtrain_m2_dit.py

# =============================================================================
# Reduced Test-Train / external-to-internal protocol
#
# Query:
#   QUERY_BASE / hospital / organ / class / image
# Support:
#   one internal/train class-folder root per organ
#
# Candidate classes are STRICT query/support intersection classes only.
# =============================================================================
QUERY_BASE=${QUERY_BASE:-${ROOT}/fsl_multi_vae/data/data_tsy_827_doctor_test_split/indomain_support_76}
HOSPITALS=(华西 自贡 苍溪 南充)
ORGANS=(食管 胃 十二指肠)

SUPPORT_ESOPHAGUS=${SUPPORT_ESOPHAGUS:-${ROOT}/fsl_multi_vae/data/data_tsy_827_doctor_test_split/食管_1query/train}
SUPPORT_STOMACH=${SUPPORT_STOMACH:-${ROOT}/fsl_multi_vae/data/data_tsy_827_doctor_test_split/胃_1query/train}
SUPPORT_DUODENUM=${SUPPORT_DUODENUM:-${ROOT}/fsl_multi_vae/data/data_tsy_827_doctor_test_split/十二指肠_1query/train}

# =============================================================================
# F1-v5 checkpoints.
# If you currently have only one ImageNet-pretrained F1, all three can point to it.
# Later, organ-specific F1 checkpoints can be supplied independently.
# =============================================================================
F1_DEFAULT=${F1_DEFAULT:-${ROOT}/weights_metric_v1/f1_v5_support_lstm_imagenet_pretrain_ddp_batch/Imagenet/step_to_support/f1_dynamic_decay_best.pt}
# F1_ESOPHAGUS=${F1_ESOPHAGUS:-${F1_DEFAULT}}
# F1_STOMACH=${F1_STOMACH:-${F1_DEFAULT}}
# F1_DUODENUM=${F1_DUODENUM:-${F1_DEFAULT}}
F1_ESOPHAGUS=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/f1_v5_support_lstm_medical/test-test/食管/step_to_support/f1_dynamic_decay_best.pt
F1_STOMACH=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/f1_v5_support_lstm_medical/test-test/胃/step_to_support/f1_dynamic_decay_best.pt
F1_DUODENUM=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/f1_v5_support_lstm_medical/test-test/十二指肠/step_to_support/f1_dynamic_decay_best.pt


# =============================================================================
# Downstream M2 + DiT/Flow RELOADS.
# Match the previous reduced/intersection no-F1 Test-Train protocol by default:
# all three organs use m2_flow_aug_m0b2 unless explicitly overridden.
# Load order: F1-v5 -> M2 reload -> RAE -> DiT/Flow reload.
# =============================================================================
M2_ESOPHAGUS=${M2_ESOPHAGUS:-${ROOT}/weights_metric_v1/m2_食管-test-train/metric_direct_episode.pt}
M2_STOMACH=${M2_STOMACH:-${ROOT}/weights_metric_v1/m2_胃-test-train/metric_direct_episode.pt}
M2_DUODENUM=${M2_DUODENUM:-${ROOT}/weights_metric_v1/m2_十二指肠-test-train/metric_direct_episode.pt}


PROJECT_ROOT=${PROJECT_ROOT:-${ROOT}}
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm_dit_L_step_265000.pt}
TRAJECTORY_SCORE_MODE=${TRAJECTORY_SCORE_MODE:-}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/fsl_multi_vae/rae_flow_metric_v1/f1_v5_buwei_eval/f1-test-test-m2_aligned-test-train-new}
EPISODES=${EPISODES:-100}
SEED=${SEED:-42}
MAX_N_WAY=${MAX_N_WAY:-8}
N_SHOT=${N_SHOT:-5}
N_QUERY=${N_QUERY:-1}
FIXED_DECAY=${FIXED_DECAY:-0.90}

# Engineering batches only; do not alter episode semantics.
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-4}
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-16}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-32}

OUTPUT_JSON=${OUTPUT_JSON:-${OUTPUT_DIR}/summary_seed${SEED}.json}

mkdir -p "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

for p in \
    "${PY}" \
    "${QUERY_BASE}" \
    "${SUPPORT_ESOPHAGUS}" \
    "${SUPPORT_STOMACH}" \
    "${SUPPORT_DUODENUM}" \
    "${F1_ESOPHAGUS}" \
    "${F1_STOMACH}" \
    "${F1_DUODENUM}" \
    "${M2_ESOPHAGUS}" \
    "${M2_STOMACH}" \
    "${M2_DUODENUM}" \
    "${DIT_CHECKPOINT}"; do
    [[ -e "${p}" ]] || { echo "[ERROR] missing: ${p}" >&2; exit 2; }
done

EXTRA_ARGS=()
if [[ -n "${TRAJECTORY_SCORE_MODE}" ]]; then
    EXTRA_ARGS+=(--trajectory_score_mode "${TRAJECTORY_SCORE_MODE}")
fi

echo "===================================================================================================================="
echo "[F1-v5 PRETRAIN CHECKPOINT EVAL / REDUCED TEST-TRAIN / EXTERNAL-TO-INTERNAL]"
echo "PY                        = ${PY}"
echo "QUERY_BASE                = ${QUERY_BASE}"
echo "HOSPITALS                 = ${HOSPITALS[*]}"
echo "ORGANS                    = ${ORGANS[*]}"
echo "SUPPORT_ESOPHAGUS         = ${SUPPORT_ESOPHAGUS}"
echo "SUPPORT_STOMACH           = ${SUPPORT_STOMACH}"
echo "SUPPORT_DUODENUM          = ${SUPPORT_DUODENUM}"
echo "F1_ESOPHAGUS              = ${F1_ESOPHAGUS}"
echo "F1_STOMACH                = ${F1_STOMACH}"
echo "F1_DUODENUM               = ${F1_DUODENUM}"
echo "M2_ESOPHAGUS              = ${M2_ESOPHAGUS}"
echo "M2_STOMACH                = ${M2_STOMACH}"
echo "M2_DUODENUM               = ${M2_DUODENUM}"
echo "DIT_CHECKPOINT            = ${DIT_CHECKPOINT}"
echo "LOAD ORDER                = F1-v5 -> M2 reload -> RAE -> DiT/Flow reload"
echo "EPISODES                  = ${EPISODES} / dataset"
echo "MAX_N_WAY / N_SHOT/Q      = ${MAX_N_WAY} / ${N_SHOT} / ${N_QUERY}"
echo "CANDIDATE SPACE           = strict query/support INTERSECTION"
echo "EPISODE_BATCH_SIZE        = ${EPISODE_BATCH_SIZE}"
echo "PAIR_BATCH_SIZE           = ${PAIR_BATCH_SIZE}"
echo "RAE_CHUNK_SIZE            = ${RAE_CHUNK_SIZE}"
echo "FIXED_DECAY diagnostic    = ${FIXED_DECAY}"
echo "OUTPUT_JSON               = ${OUTPUT_JSON}"
echo ""
echo "Main score: F1-v5 dynamic gamma -> monotonic state weights -> weighted pair score -> K-shot soft-min"
echo "Diagnostics: Direct / Endpoint / Uniform / Fixed-decay / Rescue-Degradation / gamma_t / w_t"
echo "Relative exploration: OFF during evaluation"
echo "===================================================================================================================="

python "${PY}" \
    --query_base "${QUERY_BASE}" \
    --hospitals "${HOSPITALS[@]}" \
    --expected_hospitals 4 \
    --organs "${ORGANS[@]}" \
    --expected_organs 3 \
    --organ_support \
        "食管=${SUPPORT_ESOPHAGUS}" \
        "胃=${SUPPORT_STOMACH}" \
        "十二指肠=${SUPPORT_DUODENUM}" \
    --organ_f1_checkpoint \
        "食管=${F1_ESOPHAGUS}" \
        "胃=${F1_STOMACH}" \
        "十二指肠=${F1_DUODENUM}" \
    --organ_m2_checkpoint \
        "食管=${M2_ESOPHAGUS}" \
        "胃=${M2_STOMACH}" \
        "十二指肠=${M2_DUODENUM}" \
    --project_root "${PROJECT_ROOT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --max_n_way "${MAX_N_WAY}" \
    --n_shot "${N_SHOT}" \
    --n_query "${N_QUERY}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --pair_batch_size "${PAIR_BATCH_SIZE}" \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --fixed_decay "${FIXED_DECAY}" \
    --metric_device cuda:0 \
    --flow_device cuda:0 \
    --rae_device cuda:0 \
    --output_json "${OUTPUT_JSON}" \
    --no-save_episode_details \
    --log_cuda_memory \
    "${EXTRA_ARGS[@]}"
