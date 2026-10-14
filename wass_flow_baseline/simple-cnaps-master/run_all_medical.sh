#!/bin/bash
# Serial Simple CNAPS evaluation on three medical subsets, n_runs=100 (quick smoke run).
# Forces single GPU (CUDA_VISIBLE_DEVICES=0) so it doesn't fight other jobs on shared A100s.
set -euo pipefail

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/simple-cnaps-src

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/_env_stubs:${PYTHONPATH:-}
export META_DATASET_ROOT=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/meta-dataset
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

DATA_PATH=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records
CKPT=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt
OUTDIR=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/results_medical

mkdir -p "$OUTDIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] >>> SimpleCNAPS medical eval starting (CUDA_VISIBLE_DEVICES=0)"

for subset in 食管_1query 胃_1query 十二指肠_1query; do
  case "$subset" in
    十二指肠_1query) way=7 ;;
    *)               way=8 ;;
  esac
  out="$OUTDIR/${subset}.json"
  log="$OUTDIR/${subset}.log"
  echo "[$(ts)] >>> Running $subset (way=$way, n_runs=100)"
  python run_medical_cnaps.py \
      --data_path "$DATA_PATH" \
      --subset "$subset" \
      --way "$way" \
      --shot 5 \
      --query 1 \
      --n_runs 100 \
      --test_model_path "$CKPT" \
      --feature_adaptation film \
      --output_json "$out" 2>&1 | tee "$log"
  echo "[$(ts)] <<< Finished $subset -> $out"
done

echo "[$(ts)] >>> All three subsets complete."
