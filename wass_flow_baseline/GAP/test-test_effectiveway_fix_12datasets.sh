#!/bin/bash
#SBATCH -J GAP_test_test_12
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -o pipefail

PYTHON="/mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/test-test_effectiveway_fix.py"
DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/data/data_tsy_827_doctor_test_split/indomain_support_76"
CHECKPOINT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/checkpoints/miniImageNet/4conv_8way_5shot/GAP/approx=False/GAP.outer_lr1=0.0001.outer_lr2=0.0001.batch_size=8.flags=best.th"

# Keep evaluation aligned with the 8-way 5-shot checkpoint/protocol.
SHOTS=5
QUERY_SHOTS=1
EPISODES=100

HOSPITALS=("华西" "自贡" "南充" "苍溪")
ORGANS=("食管" "胃" "十二指肠")

declare -A EPISODE_WAY
EPISODE_WAY["食管"]=8
EPISODE_WAY["胃"]=8
EPISODE_WAY["十二指肠"]=7

TOTAL=0
SUCCESS=0
FAILED=0

for HOSPITAL in "${HOSPITALS[@]}"; do
    for ORGAN in "${ORGANS[@]}"; do
        TOTAL=$((TOTAL + 1))
        TEST_DIR="${DATA_ROOT}/${HOSPITAL}/${ORGAN}"
        WAY="${EPISODE_WAY[$ORGAN]}"

        echo "============================================================"
        echo "[${TOTAL}/12] ${HOSPITAL} + ${ORGAN}"
        echo "test_dir           = ${TEST_DIR}"
        echo "nominal_episode_way= ${WAY}"
        echo "model_output_way   = 8 (fixed)"
        echo "shots/query        = ${SHOTS}/${QUERY_SHOTS}"
        echo "NOTE: Python will derive effective_way from classes with >= $((SHOTS + QUERY_SHOTS)) distinct images"
        echo "============================================================"

        if [ ! -d "${TEST_DIR}" ]; then
            echo "[SKIP] missing directory: ${TEST_DIR}"
            FAILED=$((FAILED + 1))
            echo
            continue
        fi

        python "${PYTHON}" \
            --test_dir "${TEST_DIR}" \
            --checkpoint "${CHECKPOINT}" \
            --train_way 8 \
            --episode_way "${WAY}" \
            --shots "${SHOTS}" \
            --query_shots "${QUERY_SHOTS}" \
            --episodes "${EPISODES}" \
            --GAP \
            --use_cuda

        STATUS=$?
        if [ ${STATUS} -eq 0 ]; then
            echo "[OK] ${HOSPITAL} + ${ORGAN}"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "[FAILED] ${HOSPITAL} + ${ORGAN}, exit=${STATUS}"
            FAILED=$((FAILED + 1))
        fi
        echo
    done
done

echo "============================================================"
echo "Finished: total=${TOTAL}, success=${SUCCESS}, failed=${FAILED}"
echo "============================================================"

if [ ${FAILED} -ne 0 ]; then
    exit 1
fi
