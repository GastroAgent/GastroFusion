# Wasserstein-GastroFlow Baselines

This directory collects the few-shot learning (FSL) and long-tail recognition baseline methods used in this study. It provides an index of the training and testing entry points for each method, together with evaluation notes.

## 1. Experimental Setup

Depending on their original implementations and adaptation strategies, the baselines were trained, fine-tuned, or evaluated using previously trained weights with the corresponding medical data splits from this study. They were not compared using unadapted models by default.

- **FSL baselines:** Evaluated on stomach, esophagus, and duodenum data under the N-way K-shot episodic protocols specified in the manuscript, distinguishing between test-pool support and training-pool support.
- **Long-tail recognition baselines:** Trained and evaluated using the non-episodic classification protocols corresponding to each method. These serve as contextual comparisons rather than experiments conducted under exactly the same episodic protocols as FSL.
- **Data separation:** Test query images must not be used to optimize model parameters. Test-pool support samples are reference examples provided at inference time and should not be interpreted as training on the entire test pool.

Training implementations, initialization weights, and checkpoints may differ across methods. Refer to the scripts and configurations in the respective subdirectories for details.

## 2. Training and Testing Entry Points

The table below lists the currently confirmed scripts. Some scripts use the stomach or duodenum dataset as an example; scripts and paths for other organs should be configured in the corresponding method directories.

| Category | Method | Training entry point | Testing entry point |
|---|---|---|---|
| FSL | DeepEMD | `DeepEMD/train_meta1_stomach_tsy.py` | `DeepEMD/eval_1_batch_dynamic_way_tsy_test_test.sh` |
| FSL | FSL-OM (Oblique Manifold) | `FSL-OM-main/train.py` | `FSL-OM-main/eval_test_test.sh` |
| FSL | PMF | `pmf_cvpr22/main_胃.py` | `pmf_cvpr22/test_5shot_our_new_data_test_test.py` |
| FSL | Prototypical Networks | `prototypical-network-pytorch/train_our.py` | `prototypical-network-pytorch/test_our_test_test.py` |
| FSL | PT-MAP | `PT-MAP-master/train.py` | `PT-MAP-master/test_standard_tsy.py` |
| FSL | SimpleShot | No separate training; uses PT-MAP pretrained weights | `simple_shot-master/eval_medical_tsy.sh` |
| FSL | Simple-CNAPS | `simple-cnaps-master/simple-cnaps-src/run_simple_cnaps.py` | `simple-cnaps-master/simple-cnaps-src/run_medical_cnaps_tsy.sh` |
| FSL | GAP | `GAP/main.sh` | `GAP/test-test_effectiveway_fix_12datasets.sh` |
| Long-tail | BPaCo | `BPaCo-main/sh/train_gastric_tsy.sh` | `BPaCo-main/sh/test_gastric_tsy_胃.sh` |
| Long-tail | Decoupling | `classifier-balancing-main/main.py` (or the corresponding organ-specific training script) | `classifier-balancing-main/sh/test_胃.sh` (example) |
| Long-tail | LDAM-DRW | `LDAM-DRW-master/sh/train_十二指肠_ldam.sh` | `LDAM-DRW-master/sh/test_十二指肠_ldam.sh` |

## 3. Trained Weights and Evaluation

- **FSL-OM and PT-MAP:** These methods have already been trained on the corresponding medical datasets. Evaluation loads the medical experiment checkpoints stored in their respective directories (e.g., `FSL-OM-main/checkpoints/medical/`). Retraining is not required to run evaluations using these existing checkpoints.
- **SimpleShot:** Reuses PT-MAP pretrained weights and does not perform separate model training.
- **Other methods:** Generate or prepare the required checkpoints using the training entry points listed above, then evaluate them using the corresponding testing entry points. 



> This document serves as an index of entry points and does not guarantee that every script can be executed without first updating paths and environment settings. For complete environment specifications, parameters, and organ-specific commands, consult the implementation files for each method.
