# wass_flow — RAE–Flow–SK-Wasserstein Few-Shot Classification

This implementation decouples the training pipeline into stages. The final task is **N-way K-shot episodic few-shot classification**, where support and query samples are drawn from candidate classes for each episode.

## 1. Main Pipeline

```text
query / K-shot support
→ frozen RAE (encodes endoscopic images into spatial feature maps for subsequent OT)
→ candidate support-conditioned frozen DiT Flow (generates trajectory z_0 … z_T)
→ each predicted state is mapped into metric token space by the same shared Compressor
→ SK node masses are computed once for the Direct pair and reused along the trajectory
  (F1; Q2L evaluation instead uses dynamically computed masses at each step)
→ cosine-equivalent ground cost + log-domain Sinkhorn
→ F1 LSTM decay head → monotonic state weights w_t → trajectory-weighted score
→ K-shot soft-min → N-way logits
```

**Dynamic SK:** Node masses are predicted from the data rather than assigned uniformly. The ground cost is the squared Euclidean distance between L2-normalized tokens (monotonically equivalent to cosine distance). Entropic Sinkhorn is computed in the log domain for numerical stability and returns a **regularized transport cost, not the exact W2 distance** (see `models/sk_sinkhorn.py`).

## 2. Training Stages

The metric learning stages run sequentially: **M0-A → M0-B → M1 → M2**. DiT Flow is trained **independently and in parallel** with the metric stages; neither training branch depends on the other. F1 (trajectory classification with frozen Flow) is the current primary training pipeline.

### M0-A: Classification Warm-Up

Standard classification batches, without SK, OT, Flow, episodes, or VISReg.

```bash
torchrun --nproc_per_node=8 train_m0a_cls.py \
  --train_json_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet/*.json" \
  --project_root /path/to/wass_flow \
  --output_dir /path/to/weights_metric_v1/m0a \
  --batch_size 512 --rae_chunk_size 512 --epochs 10
```

Alternatively, `--train_root / --val_root` can read ImageNet directories directly (instead of `--train_json_glob`).

Output:

```text
metric_cls_pretrain.pt
```

### M0-B: ImageNet SK-Wasserstein Metric Pretraining

Uses P×K class-balanced batches, sampling one positive and one negative for each anchor:

```bash
torchrun --nproc_per_node=8 train_m0b_ot.py \
  --train_json_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet_huaxi/*.json" \
  --project_root /path/to/wass_flow \
  --init_checkpoint /path/to/weights_metric_v1/m0a/metric_cls_pretrain.pt \
  --output_dir /path/to/weights_metric_v1/m0b \
  --classes_per_batch 50 --samples_per_class 8 \
  --batches_per_epoch 2000 --epochs 5 \
  --ot_pool_factor 1
```

Loss:

```text
OT ranking + decaying CE anchor + VISReg
```

Output:

```text
metric_imagenet_ot.pt
```

M0-B and M1 use a P×K class-balanced sampler. RAE encoding is chunked using `--rae_chunk_size` (default: 8 images per encoding call) to avoid GPU memory exhaustion with large P×K batches.

### M1: Medical Metric Adaptation

Uses the same training framework as M0-B, but switches to the medical class pool:

```bash
torchrun --nproc_per_node=8 train_m1_medical_metric.py \
  --train_json_glob "/path/to/data_train_pool_huaxi/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet_huaxi/*.json" \
  --project_root /path/to/wass_flow \
  --init_checkpoint /path/to/weights_metric_v1/m0b/metric_imagenet_ot.pt \
  --output_dir /path/to/weights_metric_v1/m1 \
  --classes_per_batch 16 --samples_per_class 8 \
  --epochs 15 --lr 3e-5
```

Output:

```text
metric_medical_adapt.pt
```

The JSON loader accepts common fields:

```text
path / image_path / x1_path / x0_path
label / label_B / class_name / category / caption
class_id / label_id / label_B_id
```

If your JSON schema differs, modify `_iter_json_items()` or the candidate field lists in `data/class_pool.py`.

### M2: Direct Episodic Calibration (First Use of Episodes)

Introduces N-way K-shot episodes for the first time, **without Flow**. The current entry point is `train_m2_direct_episode_visreg.py` (episodes + VISReg). Episodes can be sampled online from a class pool specified by `--train_json_glob` or supplied using an episode manifest JSON (the approach used in the current `.sh` files):

```bash
torchrun --nproc_per_node=8 train_m2_direct_episode_visreg.py \
  --train_episode_manifest /path/to/fsl_multi_vae/data/食管/train.json \
  --val_episode_manifest /path/to/fsl_multi_vae/data/食管/val-val.json \
  --project_root /path/to/wass_flow \
  --init_checkpoint /path/to/weights_metric_v1/m1/<m1_checkpoint>.pt \
  --output_dir /path/to/weights_metric_v1/m2_食管-test-test \
  --n_way 8 --n_shot 3 --n_query 1 \
  --episodes_per_epoch 137 --val_episodes 193 --epochs 50 \
  --visreg_weight 1e-3 --amp
```

M2 **must** initialize from an M1 checkpoint. Organ-specific pipelines initialize from the corresponding organ's M1 checkpoint. The outputs are `metric_direct_episode.pt` and `_last.pt`.

Alternative entry points: `train_m2_direct_episode_visreg_trainjson_valmanifest.py` (training class pool + validation manifest), `train_m2_direct_episode_visreg_globalcls_truebatch_v2_valsingle.py`, and `train_m2_direct_episode_flow_aug_visreg.py` (Flow trajectory augmentation).

### DiT Flow (Parallel to Metric Training)

DiT Flow training is independent of and can run in parallel with M0–M2. The training script is `train_dit_flow_spatial_sinkhorn_strict_intra_ot.py` (SLURM entry point: `train_dit_flow_spatial_sinkhorn_strict_intra_ot.sh`). **Training uses 8 GPUs; validation is an exception.**

Conditioning: the timestep and target/support image CLS features are fed to AdaLN; patch tokens are fed to Cross-Attention; and a weak noisy grayscale hint is incorporated via Spatial Addition. Caption and class conditioning are disabled by default. Hint defaults: `alpha = 0.10` and `hint_dropout = 0.10`. Images use resolution 256, aligned with trajectory generation and the metric pipeline.

```bash
torchrun --nproc_per_node=8 --master_port=29515 \
  train_dit_flow_spatial_sinkhorn_strict_intra_ot.py \
  --model otcfm \
  --dit_variant L \
  --batch_size 14 \
  --total_steps 500000 \
  --image_size 256 \
  --caption_mode vision_only \
  --no-use_class_condition \
  --no-train_text_encoder \
  --train_vision_encoder \
  --use_cross_attention \
  --hint_signal_ratio 0.10 \
  --hint_dropout 0.10 \
  --op_match_batch \
  --data_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --ot_reg_weight 0.01 --ot_reg 0.1 --ot_reg_iters 20 --ot_max_tokens 256 \
  --output_dir /path/to/weights_dit_flow_v1/vision_hint_inclass
```

The script also retains Flow Matching, OT pairing, optional OT regularization, GAN, DDP, EMA, and checkpoint cleanup logic. Full OT regularization settings are provided in the `.sh` file.

### F1: Trajectory Classification with Frozen Flow (Current Main Pipeline)

F1 is the classification branch that **uses Flow trajectories**. On top of frozen RAE, M2 MetricEncoder, and DiT Flow, it trains only a lightweight support-conditioned causal LSTM decay head (`TemporalDecayHead`):

- **Trajectory:** Generates a candidate-conditioned trajectory `z_0 … z_T` and per-step distances for each query–candidate pair.
- **Decay head:** For each transition, the LSTM predicts an information-retention factor `γ_t = γ_min + (1-γ_min)·σ(a_t)`. State weights are formed through cumulative retention: `u_0=1, u_t=∏γ_i, w_t=u_t/Σu_j`. Therefore `w_0 ≥ w_1 ≥ … ≥ w_T > 0` and `Σ w_t = 1`. This enforces monotonically non-increasing weights, encoding a prior that later trajectory states carry weaker discriminative signals, while allowing the decay rate to adapt along the trajectory.
- **Trajectory score:** `step_to_support` (`Σ w_t D(z_t, support)`) or `step_to_last` (`Σ w_t D(z_t, z_T)`), computed before the frozen M2 K-shot soft-min. The Direct baseline remains `D(z_0, support)`.
- **Optimization:** The main objective is differentiable final classification cross-entropy (CE), supplemented by antithetic group-relative perturbations for contrastive exploration.

Training proceeds in two steps (torchrun DDP):

1. **ImageNet/class-JSON pretraining:** `pretrain_f1_support_conditioned_lstm_v5_classjson_torchrun_ddp_batch.py`. Submit with `submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh` via `sbatch`; `srun` executes `run_pretrain_f1_support_conditioned_lstm_v5_multinode.sh` on each node (default: 8 processes per node, with multi-node torchrun configured by `node_rank`).
2. **Organ-specific medical training:** `train_f1_support_conditioned_lstm_v5_medical_manifest.*`. Initialize from the matching organ-specific M2 checkpoint (`--m2_checkpoint`), and load the pretrained decay head from step 1 using `PRETRAIN_F1_CHECKPOINT` / `--init_decay_checkpoint`. Use `--trajectory_score_mode step_to_support / step_to_last` to select the scoring branch.

The earlier F1 V1 implementation (trajectory-energy scoring) is historical and is no longer listed as a current entry point.

F1 trajectory generation is computationally expensive: each process holds the complete frozen DiT, EndoViT, and RAE models. Query–candidate pairs are processed in chunks rather than storing all trajectories on the GPU simultaneously.

## 3. Quick Start: Evaluation

Wasserstein-GastroFlow uses the `flow` Conda environment. Activate it before running the evaluation scripts:

```bash
conda activate flow
cd /path/to/GastroFusion/wass_flow
```

The evaluation framework supports organ-specific experiments. The publicly released Demo A is configured for stomach data under the test-test protocol. Evaluation of other organs requires the corresponding datasets, checkpoints, and configuration adjustments.

### Released Demo A: stomach test-test evaluation

Download `demo_data.zip` and `demo-wass-flow-stomach.zip` from the [GastroFusion Hugging Face repository](https://huggingface.co/GastroAgent/GastroFusion). Extract them so that the evaluation scripts can access the following default directories relative to the **GastroFusion repository root**:

```text
demo_data/stomach/<disease-class>/*.jpg
demo_weights/wass-flow-stomach/
```

The `demo_weights/wass-flow-stomach/` directory must contain the required F1/Decay, M2, Q2L, and DiT checkpoints at the paths expected by the scripts. These are extracted **directory paths**, not archive filenames. The scripts use these defaults unless `DATA_ROOT` and `WEIGHT_ROOT` are set explicitly.

Additional pretrained dependencies required by the original loaders include the RAE decoder and normalization statistics, EndoViT, and DINOv3 ViT-L/16. Ensure these are available at the configured loader paths; see the repository-level README for the expected locations.

Two released stomach evaluation entry points are provided:

| Pipeline | Script | Description |
| --- | --- | --- |
| F1 / Decay (test-test) | `eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh` | Learned decay-weighted trajectory evaluation and related diagnostic scores. |
| Q2L / Q2S-Retained (test-test) | `eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh` | Endpoint matching (`Q2Last` in code), retained-step aggregation (`Q2Mono` in code), and score-level fusion. |

```bash
# Run from GastroFusion/wass_flow after activating the flow environment.
bash eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh
bash eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh
```

To use nondefault extracted locations, set both variables before execution:

```bash
export DATA_ROOT="/absolute/path/to/stomach-class-root"
export WEIGHT_ROOT="/absolute/path/to/checkpoint-root"
```

Default output locations:

```text
wass_flow/results/f1_stomach_testtest/summary_seed42.json
wass_flow/results/q2l_stomach_testtest/fusion_q2last_q2mono/summary.csv
wass_flow/results/q2l_stomach_testtest/fusion_q2last_q2mono/summary.json
```

The released Demo A uses a default 5-shot, 1-query, 100-episode, seed-42 evaluation configuration. For detailed options and output interpretation, consult the corresponding evaluation scripts.

### Other research evaluation protocols

Additional scripts in this module cover organ-specific and test-train experiments. Their dataset layouts, checkpoint paths, and configuration may differ from the released stomach Demo A. The associated scripts and Python configuration should be checked before use; they are not interchangeable with the two public Demo A entry points.

See §4 for the geometric score definitions and evaluation protocols.

## 4. Evaluation Protocols

### Two Protocols

- **test-test:** Query and support are drawn from the same test split.
- **reduced test-train:** Candidate classes are the intersection of query-side and support-side classes (support-only classes are excluded). For shared classes with fewer than K samples, support examples are sampled with replacement.

### Geometric Scores (Same Episode, MetricEncoder / SK Geometry, and K-Shot Aggregation)

The Q2L evaluation reports two scores (`Q2S-Retained` is the name used in the manuscript and corresponds to `Q2Mono` in the code):

- **Q2Last:** Query-to-trajectory-endpoint `z_T` using Query-to-Step Dynamic-SK.
- **Q2S-Retained (retained-step mean):** Along the trajectory, retain only distances strictly greater than the previously retained distance (running maximum); average the retained distances and discard the others.

The `scorefuse_maxmin` variant combines the Q2Last / Q2S-Retained class-level distance vectors before prediction, using one of three fusion rules:

```text
dist_sum      Sum class-wise distances, then take argmin
soft_vote     Apply softmax(-d/τ) to each score, sum soft votes, then take argmax
max_then_min  For each class, take the worst distance across scores;
              choose the class with the smallest worst-case distance (max-min)
```

## 5. Key Files

```text
models/metric_encoder.py       Shared token projector
models/sk_sinkhorn.py          Dynamic SK masses + cosine Sinkhorn
                               (fixed-mass API supports "fixed SK across trajectory")
models/trajectory_scorer.py    D0/DT/area/monotonic energy + shot soft-min
models/v1_model.py             Core interfaces for Direct and full Flow
losss/visreg.py                VISReg
losss/geometry.py              Preserves the temporal-change profile of RAE/metric features
flow_adapter.py                Frozen vision+hint DiT Heun/Euler adapter
train_dit_flow_spatial_sinkhorn_strict_intra_ot.py   DiT Flow training script
```
