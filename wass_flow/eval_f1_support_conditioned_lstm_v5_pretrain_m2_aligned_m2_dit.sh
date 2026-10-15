#!/bin/bash
#SBATCH -J eval_f1_v5_nodirect_4x3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.err

# export CUDA_VISIBLE_DEVICES='0'

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.py

# =============================================================================
# Test-test data layout, same style as eval_no_f1...m2_aligned.py:
#   DATA_BASE / hospital / organ / class / image
# Query and support are sampled from the same hospital-organ pool, while the
# exact query image is excluded from support.
# =============================================================================
DATA_BASE=${DATA_BASE:-${ROOT}/fsl_multi_vae/data/data_tsy_827_doctor_test_split/indomain_support_76}
HOSPITALS=(自贡 苍溪 南充)
ORGANS=(食管 胃 十二指肠)
# export CUDA_VISIBLE_DEVICES=5
# =============================================================================
# F1-v5 checkpoints.
#
# Preferred usage: one ImageNet-pretrained F1 per downstream organ/M2.  If you
# currently trained only ONE generic ImageNet F1, simply leave all three pointing
# to the same checkpoint.  The checkpoint already contains its frozen MetricEncoder,
# shot_tau/cls_tau, TemporalDecayHead, and training configuration.
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
# Load order in Python:
#   1) F1-v5 head/config
#   2) organ-specific M2 MetricEncoder + shot_tau/cls_tau
#   3) RAE
#   4) explicit DiT/Flow checkpoint
# The M2/DiT states stored inside the F1 pretraining bundle are NOT final.
# =============================================================================
M2_ESOPHAGUS=${M2_ESOPHAGUS:-${ROOT}/weights_metric_v1/new_m2_食管-test-test/metric_direct_episode.pt}
M2_STOMACH=${M2_STOMACH:-${ROOT}/weights_metric_v1/new_m2_胃-test-test/metric_direct_episode.pt}
M2_DUODENUM=${M2_DUODENUM:-${ROOT}/weights_metric_v1/new_m2_十二指肠-test-test/metric_direct_episode.pt}

PROJECT_ROOT=${PROJECT_ROOT:-${ROOT}}
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt}
TRAJECTORY_SCORE_MODE=${TRAJECTORY_SCORE_MODE:-}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/fsl_multi_vae/rae_flow_metric_v1/f1_v5_buwei_eval/f1-test-test-m2_aligned-test-test-nodirect}
EPISODES=${EPISODES:-100}
SEED=${SEED:-42}
MAX_N_WAY=${MAX_N_WAY:-8}
N_SHOT=${N_SHOT:-5}
N_QUERY=${N_QUERY:-1}
FIXED_DECAY=${FIXED_DECAY:-0.90}

# Engineering batches only; they do NOT change N-way/K-shot episode semantics.
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-4}
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-16}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-32}

OUTPUT_JSON=${OUTPUT_JSON:-${OUTPUT_DIR}/summary_seed${SEED}.json}

mkdir -p "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

for p in \
    "${PY}" "${DATA_BASE}" \
    "${F1_ESOPHAGUS}" "${F1_STOMACH}" "${F1_DUODENUM}" \
    "${M2_ESOPHAGUS}" "${M2_STOMACH}" "${M2_DUODENUM}" \
    "${DIT_CHECKPOINT}"; do
    [[ -e "${p}" ]] || { echo "[ERROR] missing: ${p}" >&2; exit 2; }
done

EXTRA_ARGS=()
if [[ -n "${TRAJECTORY_SCORE_MODE}" ]]; then
    EXTRA_ARGS+=(--trajectory_score_mode "${TRAJECTORY_SCORE_MODE}")
fi

echo "===================================================================================================================="
echo "[F1-v5 PRETRAIN CHECKPOINT EVAL / M2-ALIGNED TEST-TEST / NO-DIRECT REPORTING]"
echo "PY                        = ${PY}"
echo "DATA_BASE                 = ${DATA_BASE}"
echo "HOSPITALS                 = ${HOSPITALS[*]}"
echo "ORGANS                    = ${ORGANS[*]}"
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
echo "EPISODE_BATCH_SIZE        = ${EPISODE_BATCH_SIZE}"
echo "PAIR_BATCH_SIZE           = ${PAIR_BATCH_SIZE}"
echo "RAE_CHUNK_SIZE            = ${RAE_CHUNK_SIZE}"
echo "FIXED_DECAY diagnostic    = ${FIXED_DECAY}"
echo "OUTPUT_JSON               = ${OUTPUT_JSON}"
echo ""
echo "Main score: learned F1-v5 dynamic gamma -> monotonic state weights -> weighted pair score -> K-shot soft-min"
echo "Diagnostics: Endpoint / Uniform / Fixed-decay / gamma_t / w_t (Direct not reported)"
echo "Relative exploration: OFF during evaluation"
echo "===================================================================================================================="

python "${PY}" \
    --batch_root "${DATA_BASE}" \
    --hospitals "${HOSPITALS[@]}" \
    --expected_hospitals 3 \
    --organs "${ORGANS[@]}" \
    --expected_organs 3 \
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
