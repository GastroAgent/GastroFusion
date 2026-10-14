#!/bin/bash

# ============================================================================
# DeepEMD Episodic Evaluation Scripts
# ============================================================================

# 设置默认参数
DATA_DIR='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/data/十二指肠_1query'
MODEL_DIR='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/十二指肠/fcn/十二指肠-5shot-7way/max_acc.pth'
GPU='0'

# ============================================================================
# 1. 运行 eposide 风格的 Sinkhorn W2 评估
# ============================================================================
run_eposide_style() {
    echo "========================================"
    echo "Running eposide-style evaluation (Sinkhorn W2)"
    echo "========================================"
    echo "Mode: ${EVAL_MODE:-sinkhorn}"
    echo "Way: ${WAY:-7}, Shot: ${SHOT:-5}"
    echo "Episodes: ${EPISODES:-100}"
    echo "Sinkhorn epsilon: ${SINKHORN_EPSILON:-0.05}"
    echo "Sinkhorn max_points: ${SINKHORN_MAX_POINTS:-4096}"
    echo "Support aggregation: ${SUPPORT_AGG:-min}"
    echo ""

    python eval_eposide_style.py \
        -data_dir "${DATA_DIR}" \
        -model_dir "${MODEL_DIR}" \
        -way "${WAY:-7}" \
        -shot "${SHOT:-5}" \
        -query 1 \
        -test_episode "${EPISODES:-100}" \
        -gpu "${GPU}" \
        -seed "${SEED:-1}" \
        -eval_mode "${EVAL_MODE:-sinkhorn}" \
        -sinkhorn_epsilon "${SINKHORN_EPSILON:-0.05}" \
        -sinkhorn_max_points "${SINKHORN_MAX_POINTS:-4096}" \
        -support_agg "${SUPPORT_AGG:-min}" \
        -compare_with_emd "${COMPARE_WITH_EMD:-True}" \
        -save_dir "./sinkhorn_results_${EVAL_MODE:-sinkhorn}_$(date +%Y%m%d_%H%M%S)"
}

# ============================================================================
# 2. 运行混合模式评估
# ============================================================================
run_hybrid() {
    echo "========================================"
    echo "Running hybrid evaluation (EMD + Sinkhorn W2)"
    echo "========================================"
    echo "Distance weight (Sinkhorn): ${DISTANCE_WEIGHT:-0.5}"
    echo ""

    python eval_eposide_style.py \
        -data_dir "${DATA_DIR}" \
        -model_dir "${MODEL_DIR}" \
        -way "${WAY:-7}" \
        -shot "${SHOT:-5}" \
        -query 1 \
        -test_episode "${EPISODES:-100}" \
        -gpu "${GPU}" \
        -seed "${SEED:-1}" \
        -eval_mode hybrid \
        -distance_weight "${DISTANCE_WEIGHT:-0.5}" \
        -sinkhorn_epsilon "${SINKHORN_EPSILON:-0.05}" \
        -sinkhorn_max_points "${SINKHORN_MAX_POINTS:-4096}" \
        -support_agg "${SUPPORT_AGG:-min}" \
        -compare_with_emd True \
        -save_dir "./hybrid_results_$(date +%Y%m%d_%H%M%S)"
}

# ============================================================================
# 3. 运行原始 DeepEMD 评估
# ============================================================================
run_original_emd() {
    echo "========================================"
    echo "Running original DeepEMD evaluation"
    echo "========================================"

    python eval_1.py \
        -data_dir "${DATA_DIR}" \
        -model_dir "${MODEL_DIR}" \
        -way "${WAY:-7}" \
        -shot "${SHOT:-5}" \
        -query 1 \
        -test_episode "${EPISODES:-100}" \
        -gpu "${GPU}" \
        -seed "${SEED:-1}" \
        -save_dir "./original_emd_results_$(date +%Y%m%d_%H%M%S)"
}

# ============================================================================
# 4. 比较三种方法
# ============================================================================
run_comparison() {
    echo "========================================"
    echo "Running comparison: EMD vs Sinkhorn W2 vs Hybrid"
    echo "========================================"

    # 设置临时结果目录
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_DIR="./comparison_results_${TIMESTAMP}"
    mkdir -p "${RESULT_DIR}"

    echo ""
    echo ">>> 1. Running original EMD evaluation..."
    python eval_1.py \
        -data_dir "${DATA_DIR}" \
        -model_dir "${MODEL_DIR}" \
        -way "${WAY:-7}" \
        -shot "${SHOT:-5}" \
        -query 1 \
        -test_episode "${EPISODES:-100}" \
        -gpu "${GPU}" \
        -seed "${SEED:-1}" \
        -save_dir "${RESULT_DIR}/emd"

    echo ""
    echo ">>> 2. Running Sinkhorn W2 evaluation (eposide style)..."
    python eval_eposide_style.py \
        -data_dir "${DATA_DIR}" \
        -model_dir "${MODEL_DIR}" \
        -way "${WAY:-7}" \
        -shot "${SHOT:-5}" \
        -query 1 \
        -test_episode "${EPISODES:-100}" \
        -gpu "${GPU}" \
        -seed "${SEED:-1}" \
        -eval_mode sinkhorn \
        -sinkhorn_epsilon "${SINKHORN_EPSILON:-0.05}" \
        -sinkhorn_max_points "${SINKHORN_MAX_POINTS:-4096}" \
        -support_agg "${SUPPORT_AGG:-min}" \
        -compare_with_emd False \
        -save_dir "${RESULT_DIR}/sinkhorn"

    echo ""
    echo ">>> 3. Running hybrid evaluation..."
    python eval_eposide_style.py \
        -data_dir "${DATA_DIR}" \
        -model_dir "${MODEL_DIR}" \
        -way "${WAY:-7}" \
        -shot "${SHOT:-5}" \
        -query 1 \
        -test_episode "${EPISODES:-100}" \
        -gpu "${GPU}" \
        -seed "${SEED:-1}" \
        -eval_mode hybrid \
        -distance_weight "${DISTANCE_WEIGHT:-0.5}" \
        -sinkhorn_epsilon "${SINKHORN_EPSILON:-0.05}" \
        -sinkhorn_max_points "${SINKHORN_MAX_POINTS:-4096}" \
        -support_agg "${SUPPORT_AGG:-min}" \
        -compare_with_emd False \
        -save_dir "${RESULT_DIR}/hybrid"

    echo ""
    echo "========================================"
    echo "Comparison Results Saved to: ${RESULT_DIR}"
    echo "========================================"
    echo ""
    echo "To view results:"
    echo "  - EMD:        ${RESULT_DIR}/emd/"
    echo "  - Sinkhorn:   ${RESULT_DIR}/sinkhorn/"
    echo "  - Hybrid:      ${RESULT_DIR}/hybrid/"
}

# ============================================================================
# 主菜单
# ============================================================================
show_help() {
    echo ""
    echo "========================================"
    echo "DeepEMD Episodic Evaluation Scripts"
    echo "========================================"
    echo ""
    echo "Usage: bash run_eval.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  eposide     - Run eposide-style Sinkhorn W2 evaluation"
    echo "  hybrid      - Run hybrid evaluation (EMD + Sinkhorn)"
    echo "  original    - Run original DeepEMD evaluation"
    echo "  compare     - Compare all three methods"
    echo "  help        - Show this help message"
    echo ""
    echo "Common Options (set as environment variables):"
    echo "  WAY=7                    - Number of ways (classes per episode)"
    echo "  SHOT=5                   - Number of shots per class"
    echo "  EPISODES=100             - Number of test episodes"
    echo "  GPU=0                    - GPU device ID"
    echo "  SEED=1                   - Random seed"
    echo "  EVAL_MODE=sinkhorn        - Evaluation mode (sinkhorn/emd/hybrid)"
    echo "  SINKHORN_EPSILON=0.05    - Sinkhorn regularization parameter"
    echo "  SINKHORN_MAX_POINTS=4096 - Max points for Sinkhorn computation"
    echo "  SUPPORT_AGG=min          - Support aggregation (min/mean/median)"
    echo "  DISTANCE_WEIGHT=0.5      - Weight for Sinkhorn in hybrid mode"
    echo "  COMPARE_WITH_EMD=True     - Compare results with original EMD"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Run Sinkhorn W2 evaluation (eposide style)"
    echo "  bash run_eval.sh eposide"
    echo ""
    echo "  # Run with custom parameters"
    echo "  WAY=5 SHOT=3 EPISODES=200 bash run_eval.sh eposide"
    echo ""
    echo "  # Compare all methods"
    echo "  bash run_eval.sh compare"
    echo ""
    echo "  # Run hybrid evaluation with custom weight"
    echo "  DISTANCE_WEIGHT=0.3 bash run_eval.sh hybrid"
    echo ""
}

# ============================================================================
# 解析命令
# ============================================================================
COMMAND="${1:-help}"

case "${COMMAND}" in
    eposide)
        run_eposide_style
        ;;
    hybrid)
        run_hybrid
        ;;
    original)
        run_original_emd
        ;;
    compare)
        run_comparison
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        echo ""
        show_help
        exit 1
        ;;
esac
