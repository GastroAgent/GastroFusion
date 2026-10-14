#!/bin/bash

set -euo pipefail

# ============================================================
# 路径配置
# ============================================================

# generate_medical_csv.py 所在目录
SIMPLESHOT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple_shot-master"

# PT-MAP的JSON文件列表根目录
FILELIST_DIR="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/PT-MAP-master/filelists/medical"

# 原始图片数据根目录
DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data"

# CSV输出根目录
OUT_DIR="${SIMPLESHOT_ROOT}/split_tsy"

# ============================================================
# 待转换的数据子集
# ============================================================

SUBSETS=(
    "食管_7way_1query"
    "胃_7way_1query"
    "十二指肠_7way_1query"
    "食管_8way_1query"
)

# ============================================================
# 批量生成CSV
# ============================================================

cd "${SIMPLESHOT_ROOT}"

for SUBSET in "${SUBSETS[@]}"; do
    echo "============================================================"
    echo "Generating SimpleShot CSV: ${SUBSET}"
    echo "============================================================"

    SUBSET_DIR="${DATA_ROOT}/${SUBSET}"

    if [[ ! -d "${SUBSET_DIR}" ]]; then
        echo "[ERROR] Data directory not found:"
        echo "  ${SUBSET_DIR}"
        exit 1
    fi

    if [[ ! -d "${FILELIST_DIR}/${SUBSET}" ]]; then
        echo "[ERROR] Filelist directory not found:"
        echo "  ${FILELIST_DIR}/${SUBSET}"
        exit 1
    fi

    python generate_medical_csv.py \
        --subset "${SUBSET}" \
        --filelist_dir "${FILELIST_DIR}" \
        --subset_dir "${SUBSET_DIR}" \
        --out_dir "${OUT_DIR}"
done

echo "============================================================"
echo "All CSV files generated."
echo "Output root: ${OUT_DIR}"
echo "============================================================"