# GastroFusion — Geometric–Semantic Inference, Fusion, and Evaluation

[English](README_EN.md) | [简体中文](README_CN.md)

The `mllm_fsl/` directory applies the pretrained **GastroMLLM (Med-Qwen3-VL)** and **Wasserstein-GastroFlow** models to internal and external gastrointestinal endoscopy cohorts. It generates predictions from both branches, selects organ-specific fusion parameters using validation data, performs final fusion, and evaluates parameter sensitivity. **This directory does not train the two backbone models**; see the repository-level `mllm/` and `wass_flow/` directories for model training.

> This documentation describes the code, data, and released results in the latest repository version. Some scripts retain absolute paths from the original research server and must be adjusted before execution. Providing JSON files does not imply that the referenced original clinical images or model weights are publicly available.

## 1. Overall Pipeline

```text
                Internal WCH / external centre–organ cohorts
                                    |
                      +-------------+-------------+
                      |                           |
                 GastroMLLM                Wasserstein-GastroFlow
              Semantic evidence               Geometric evidence
                      |                           |
             data/mllm_data/ JSON         data/fsl_data/ flat JSON
                      |                           |
             MLLM inference server         RAE → Metric → Flow
                      |                           |
            semantic_score_mllm_*.py     offline_align_rae_metric_*
                      |                           |
               mllm/mllm_result/           fsl/fsl_result/
                      |                           |
                      +-------------+-------------+
                                    |
                   val_grid/ Validation-set parameter search
                                    |
                      Select and fix fusion parameters
                                    |
                        fusion/ Internal/external fusion
                                    |
                     mllm_fsl_detail/mllm_fsl_result/
                                    |
                      sensitivity/ Parameter sensitivity
```

The `val_grid/`, `fusion/`, and `sensitivity/` modules serve **different purposes**: `val_grid/` compares candidate parameter settings on independent validation data; `fusion/` produces case-level fused predictions with the selected configuration; and `sensitivity/` examines how results respond to parameter perturbations. Final test data are not used to reselect parameters.

## 2. Directory Structure

```text
mllm_fsl/
├── data/
│   ├── mllm_data/
│   │   ├── external_few_way/              # External cohort-aligned inputs (9 JSON files)
│   │   └── external_lot_way/              # External full-organ inputs (9 JSON files)
│   └── fsl_data/
│       ├── huaxi/support_831_val/         # WCH flat JSON files for three organs
│       └── external/
│           ├── mllm_fsl_subset_agent_few_way_91/  # 3 centres × 3 organs
│           └── mllm_fsl_subset_agent_lot_way_91/  # 3 centres × 3 organs
├── mllm/
│   ├── mllm_code/                         # Model serving and semantic scoring scripts
│   └── mllm_result/                       # Internal and external semantic results
├── fsl/
│   ├── fsl_code/                          # Offline RAE/Metric/Flow inference scripts
│   └── fsl_result/                        # Geometric evaluation results included in the repository
└── mllm_fsl_detail/
    ├── mllm_fsl_code/
    │   ├── val_grid/                      # Validation-set grid search
    │   ├── fusion/                        # Internal and external fusion
    │   └── sensitivity/                   # Parameter sensitivity analysis
    └── mllm_fsl_result/                   # Released final fusion results
```

**Dataset naming:** `huaxi` refers to West China Hospital (WCH). The external centres are Nanchong, Zigong, and Cangxi, each covering the esophagus, stomach, and duodenum. `few_way` denotes the external **cohort-aligned** candidate-label space (disease classes represented in a given cohort), whereas `lot_way` denotes the **full-organ** label space (the complete disease-label set for the organ). These are **candidate-label-space settings**, not different model versions.

## 3. Input Data (`data/`)

### 3.1 GastroMLLM Input (`data/mllm_data/`)

The released directory contains two external input sets, each consisting of nine `<centre><organ>_agent.json` files:

```text
data/mllm_data/
├── external_few_way/南充胃_agent.json
└── external_lot_way/南充胃_agent.json
```

The MLLM input describes the evaluation cases, image paths, and associated candidate information. `semantic_score_mllm_external.py` parses these records and sends them to the MLLM service to obtain open-ended diagnoses and semantic scores for candidate diseases.

### 3.2 Wasserstein-GastroFlow Input (`data/fsl_data/`)

```text
data/fsl_data/
├── huaxi/support_831_val/
│   ├── 食管_final_doctor_exam_flat.json
│   ├── 胃_final_doctor_exam_flat.json
│   └── 十二指肠_final_doctor_exam_flat.json
└── external/
    ├── mllm_fsl_subset_agent_few_way_91/<centre>/<organ>/final_doctor_exam_*_flat.json
    └── mllm_fsl_subset_agent_lot_way_91/<centre>/<organ>/final_doctor_exam_*_flat.json
```

Each flat JSON contains **paired data** used to construct query–support comparisons. Records include fields such as `data_id`, `question_id`, `x0`, `x1`, `label_A`, `label_B`, `support_dir`, and `support_selection`. Image paths in fields such as `x0` and `x1` must be accessible from the local file system. One query may have multiple candidate classes and multiple support pairs.

## 4. GastroMLLM Semantic Branch (`mllm/`)

| File | Purpose |
|---|---|
| `mllm_code/gastro_mllm_qwenvl3_server.py` | Loads the medically adapted Qwen3-VL model and provides semantic inference/scoring services. |
| `mllm_code/gastro-mllm-server.sh` | Configures the environment, checkpoint, and service port, then starts the server. |
| `mllm_code/semantic_score_mllm_huaxi.py` | Calls the service for WCH cases and organizes semantic scores. |
| `mllm_code/semantic_score_mllm_external.py` | Performs batch inference and semantic scoring for the nine external centre–organ cohorts. |
| `mllm_code/run_batch_mllm_huaxi.sh` | Batch launcher for WCH. |
| `mllm_code/run_batch_mllm_external.sh` | Batch launcher for external cohorts. |

First start the service, then run the relevant batch script. The server and client may be deployed on different nodes. Before execution, ensure their host, port, network accessibility, and model-weight paths are consistent.

```bash
# Run from the repository root after updating paths and service addresses.
bash mllm_fsl/mllm/mllm_code/gastro-mllm-server.sh

# In a separate terminal or job, run the appropriate cohort pipeline.
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_huaxi.sh
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_external.sh
```

**Result directories (corresponding files are included in the current repository):**

```text
mllm/mllm_result/
├── mllm_result_huaxi/<organ>/Med-qwen3-vl-RL_latent_<organ>.json
├── mllm_result_external9_few_way/<centre>/<organ>/Med-qwen3-vl-RL_latent_<centre>_<organ>.json
└── mllm_result_external9_lot_way/<centre>/<organ>/Med-qwen3-vl-RL_latent_<centre>_<organ>.json
```

Each result file contains a `results` list and, where applicable, execution settings and summary information. Case-level fields typically include `image_id`, `target_image`, `ground_truth`, `mllm_diag`, `mllm_score_map`, and `candidate_diseases`. Fusion uses **semantic scoring evidence** over candidate labels (and associated confidence measures); reference labels are used for offline evaluation.

## 5. Wasserstein-GastroFlow Geometric Branch (`fsl/`)

| File | Purpose |
|---|---|
| `fsl_code/offline_align_rae_metric_flow_service_new_test_test.py` | Computes offline RAE + Metric + Flow geometric distances for the relevant test-test setting. |
| `fsl_code/offline_align_rae_metric_flow_service_new_test_train.py` | Computes offline geometric distances for the relevant test-train setting. |
| `fsl_code/run_batch_3huaxi_5shot.sh` | Batch evaluation of fixed query–support pairs for all three WCH organs. |
| `fsl_code/run_batch_9_5shot_fewway.sh` | Batch evaluation of the nine external cohort-aligned cohorts. |
| `fsl_code/run_batch_9_5shot_lotway.sh` | Batch evaluation of the nine external full-organ cohorts. |

This branch extracts image representations, generates candidate-conditioned trajectories, and computes geometric matching distances from the supplied query–support pairs. Distances are then aggregated by class to obtain candidate-disease distances, class rankings, and final predictions. Offline evaluation saves both pair-level and query-level results so downstream programs can directly retrieve class-level distances.

```bash
# Run from the repository root after updating paths and model checkpoints.
bash mllm_fsl/fsl/fsl_code/run_batch_3huaxi_5shot.sh
bash mllm_fsl/fsl/fsl_code/run_batch_9_5shot_fewway.sh
bash mllm_fsl/fsl/fsl_code/run_batch_9_5shot_lotway.sh
```

**Common FSL output files:**

| File | Purpose |
|---|---|
| `result.json` | Original query–support pairs and associated geometric-distance information. |
| `query_results.json` | Class-level results aggregated by query; the preferred input for the final fusion program. |

**The FSL results included in this repository snapshot contain only:**

```text
fsl/fsl_result/huaxi/食管/query_results.json
```

Geometric results for the other organs and external cohorts must be generated using the corresponding scripts or supplied separately.

## 6. Validation-Grid Parameter Selection (`val_grid/`)

`mllm_fsl_detail/mllm_fsl_code/val_grid/val_grid_search.py` enumerates organ-specific parameter combinations using validation-set MLLM and FSL predictions, invokes the internal fusion Python program, and aggregates and ranks the results according to validation performance.

**Inputs:** The validation-set MLLM results root (`INPUT_ROOT`), the corresponding geometric results root (`RAE_ROOT`), and the parameter grid (`GRID`). **Outputs:** Per-configuration prediction results and summary/ranking files under `OUTPUT_ROOT`, which support selection of a fixed parameter configuration. The script's `PY`, `INPUT_ROOT`, `RAE_ROOT`, and `OUTPUT_ROOT` settings currently use research-server paths and must be adjusted for the actual validation data.

The parameter ranges include score-fusion weights and protection thresholds for the stomach, as well as semantic-override thresholds for the esophagus and duodenum. Parameters selected on validation data are then written into the corresponding Fusion Shell configuration.

## 7. Final Geometric–Semantic Fusion (`fusion/`)

| File | Purpose |
|---|---|
| `fsl_mllm_fusion_tolast_rae_v2_huaxi.py` | Case-level fusion and cohort statistics for internal WCH data. |
| `fsl_mllm_fusion_tolast_rae_v2_external.py` | Fusion and statistics for external centre–organ cohorts. |
| `fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh` | Shell / SLURM launcher for WCH fusion. |
| `fsl_mllm_fusion_tolast_rae_v2_external_grid_val.sh` | Shell / SLURM launcher for external fusion. |

**Connecting the inputs:**

- `INPUT_ROOT`: The primary MLLM result JSON files under `mllm_result_huaxi/` or `mllm_result_external9_{few,lot}_way/`. Each case contains semantic predictions and disease scores.
- `RAE_FSL_RESULT_ROOT` (passed to Python as `--rae_fsl_result_root`): The root of offline geometric results. The program locates `query_results.json` by organ or centre–organ cohort and falls back to `result.json` when necessary.
- Case identifiers, candidate-diagnosis names, and candidate-label spaces must be aligned between branches. Do not mix cohort-aligned and full-organ results.

**Fusion rules:** For the stomach, score fusion incorporates a safeguard based on the geometric matching margin. For the esophagus, the geometric prediction is used by default, with a high-confidence semantic override permitted. For the duodenum, a stricter semantic override condition is applied. The specific parameters are explicitly set in the two `.sh` scripts and are fixed for the corresponding test data after validation-set selection.

**Outputs:** Case-level fusion JSON files typically contain the original diagnostic information, FSL class-level geometric distances and rankings, predictions from both branches, the fused prediction, correctness indicators, and override diagnostics. Cohort-level JSON/CSV summaries report branch and fusion accuracies, corrected and degraded predictions, and matching failures. The released internal results are located at:

```text
mllm_fsl_detail/mllm_fsl_result/
└── result_huaxi_831_grid/v2_fusion_traj_tlast_rae_latent_tsy/
    ├── 十二指肠/Med-qwen3-vl-RL_latent_十二指肠_fusion_v2_traj.json
    ├── 胃/Med-qwen3-vl-RL_latent_胃_fusion_v2_traj.json
    └── 食管/Med-qwen3-vl-RL_latent_食管_fusion_v2_traj.json
```

In this repository snapshot, `mllm_fsl_result/` **contains only the three organ-specific case-level fusion JSON files listed above**. External fusion results and batch summary/failure reports are not included, but the corresponding scripts can generate them under the configured output directories.

```bash
# Run from the repository root after updating PY, input/output roots, and the environment.
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_external_grid_val.sh
```

## 8. Parameter Sensitivity (`sensitivity/`)

| File | Purpose |
|---|---|
| `fsl_mllm_fusion_tolast_rae_v2_huaxi_sweep.py` / `.sh` | Sensitivity experiments on internal WCH cohorts. |
| `fsl_mllm_fusion_tolast_rae_v2_sweep.py` / `.sh` | Sensitivity experiments on external cohorts. |

Sensitivity analysis perturbs selected parameters around the previously determined fusion configuration to examine changes in accuracy, corrected predictions, and degraded predictions. It provides an assessment of robustness rather than a means of reselecting parameters on an independent test set. Before execution, check the `.sh` settings for `SWEEP_PY`, input-result directories, and output roots, and verify that the fusion code used by the analysis is consistent with the current final fusion implementation.

## 9. Reproduction Workflows

### Reproducing Results from Input Data

1. Prepare the required images, MLLM input JSON files, FSL flat paired JSON files, and matching model checkpoints.
2. Start the GastroMLLM service and run internal or external semantic batch inference to generate `mllm_result/`.
3. Run offline geometric batch inference for the corresponding evaluation protocol to generate each cohort's `query_results.json` (and `result.json` where required).
4. Run `val_grid/` on independent validation data to determine organ-specific parameters.
5. Fix the selected parameters in the `fusion/` Shell configurations and perform fusion for the matched internal or external cohorts.
6. Optionally run `sensitivity/` to examine local parameter stability and changes in prediction errors.

For backbone-model training and underlying implementations, see the repository-level `mllm/README.md` and `wass_flow/README_CN.md`. This document focuses on the **data organization, inference, parameter selection, and fusion workflows under `mllm_fsl/`**.
