import argparse
import time
import torch.nn as nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from scipy import stats
from collections import defaultdict

from Models.dataloader.samplers import CategoriesSampler
from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *

DATA_DIR = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy_deepemd/cross_domain_balanced/indomain_support_76/苍溪/胃'
MODEL_DIR = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/cifar_fs/fcn/5shot-8way/max_acc.pth'
# MODEL_DIR='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/十二指肠/fcn/十二指肠-5shot-7way/max_acc.pth'
# MODEL_DIR='/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/食管/fcn/食管-5shot-8way/max_acc.pth'
parser = argparse.ArgumentParser()
# output
parser.add_argument('-save_dir', type=str, default='/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/outputs_metric_results/test-test/苍溪/胃',
                    help='Directory to save metric plots')
# about task
parser.add_argument('-way', type=int, default=2)
parser.add_argument('-shot', type=int, default=5)
parser.add_argument('-query', type=int, default=1)
parser.add_argument('-dataset', type=str, default='cifar_fs',
                    choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs'])
parser.add_argument('-set', type=str, default='test', choices=['train', 'val', 'test'])
# about model
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
parser.add_argument('-norm', type=str, default='center', choices=['center'])
parser.add_argument('-deepemd', type=str, default='sampling', choices=['fcn', 'grid', 'sampling'])
parser.add_argument('-feature_pyramid', type=str, default=None)
parser.add_argument('-num_patch', type=int, default=9)
parser.add_argument('-patch_list', type=str, default='2,3')
parser.add_argument('-patch_ratio', type=float, default=2)
# solver
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
# SFC
parser.add_argument('-sfc_lr', type=float, default=100)
parser.add_argument('-sfc_wd', type=float, default=0)
parser.add_argument('-sfc_update_step', type=float, default=100)
parser.add_argument('-sfc_bs', type=int, default=4)
# others
parser.add_argument('-test_episode', type=int, default=100)
parser.add_argument('-gpu', default='0,1,2,5,6,7')
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
parser.add_argument('-model_dir', type=str, default=MODEL_DIR)
parser.add_argument('-seed', type=int, default=1)

args = parser.parse_args()
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

pprint(vars(args))
set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset = set_up_datasets(args)
os.makedirs(args.save_dir, exist_ok=True)

# model
model = DeepEMD(args)
model = load_model(model, args.model_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# test dataset
test_set = Dataset(args.set, args)
# ===================== 正确版本：读取 test 下的类别 =====================
import os
print("\n" + "="*50)
print("            【固定类别编号 ↔ 文件夹名】")
print("="*50)

# 自动拼接：你的数据集/test
# test_folder = os.path.join(args.data_dir, 'test')
test_folder = args.data_dir
class_names = sorted([d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))])

for idx, cls_name in enumerate(class_names):
    print(f"Class {idx}   --->   {cls_name}")

print("="*50 + "\n")
# ======================================================================
sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
tqdm_gen = tqdm.tqdm(loader)

# label of query images
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)

# ------------------------------------------------------------------ #
#  Accumulators for the three extra metrics
# ------------------------------------------------------------------ #

# (1) Confusion boundary analysis
#     For every episode we store, for each query image:
#       - its true class index (0..way-1)
#       - the predicted class index
#     We accumulate a way×way confusion matrix across all episodes.
confusion_matrix_accum = np.zeros((args.way, args.way), dtype=np.int64)

# (2) Distance quality analysis
#     For each episode we compute pairwise distances between every
#     query embedding and every support prototype, then split into
#     intra-class (same label) and inter-class (different label) pools.
intra_distances = []   # list of scalars
inter_distances = []   # list of scalars

# (3) Complexity–performance trade-off
#     We record wall-clock inference time per episode.
latency_per_episode = []   # seconds

# ------------------------------------------------------------------ #
#  Main evaluation loop
# ------------------------------------------------------------------ #
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))

with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot

        # --- encoder pass (not timed for latency; only meta pass is) ---
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]

        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)

        # --- timed meta / matching pass ---
        t0 = time.perf_counter()
        model.module.mode = 'meta'
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_per_episode.append(t1 - t0)

        # --- accuracy ---
        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc

        # ---- (1) Confusion matrix ----
        preds = logits.argmax(dim=1).cpu().numpy()          # (way*query,)
        true_labels = label.cpu().numpy()                   # (way*query,)
        for t, p in zip(true_labels, preds):
            confusion_matrix_accum[t, p] += 1

        # ---- (2) Distance quality ----
        # logits shape: (way*query, way)
        # Each row is the similarity/distance score to each prototype.
        # We convert logits to distances: distance = -logit / temperature
        # (works for cosine similarity scaled by temperature)
        dist_matrix = (-logits / args.temperature).cpu().numpy()  # (n_query, way)
        for q_idx in range(dist_matrix.shape[0]):
            true_cls = true_labels[q_idx]
            for cls in range(args.way):
                d = dist_matrix[q_idx, cls]
                if cls == true_cls:
                    intra_distances.append(d)
                else:
                    inter_distances.append(d)

        m, pm = compute_confidence_interval(test_acc_record[:i])
        tqdm_gen.set_description(
            'batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))

# ------------------------------------------------------------------ #
#  Print original accuracy results
# ------------------------------------------------------------------ #
m, pm = compute_confidence_interval(test_acc_record)
result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
print(result_list[0])
print(result_list[1])

# ================================================================== #
#  METRIC 1 — Confusion Boundary Analysis
# ================================================================== #
print("\n========== Metric 1: Confusion Boundary Analysis ==========")

# Normalise confusion matrix to per-class recall
cm_norm = confusion_matrix_accum.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
cm_norm = cm_norm / (row_sums + 1e-8)

# Find the top-3 most confused pairs (off-diagonal)
off_diag = []
for r in range(args.way):
    for c in range(args.way):
        if r != c:
            off_diag.append((confusion_matrix_accum[r, c], r, c))
off_diag.sort(reverse=True)

print("Top-5 most confused class pairs (true → predicted, count):")
for count, r, c in off_diag[:5]:
    print(f"  Class {r} → Class {c}: {count} misclassifications "
          f"({count / row_sums[r, 0] * 100:.1f}% of class {r} queries)")

# Compute per-class accuracy
per_class_acc = np.diag(cm_norm) * 100
print("\nPer-class accuracy (%):")
for cls in range(args.way):
    print(f"  Class {cls}: {per_class_acc[cls]:.2f}%")

# Hard-pair accuracy: accuracy restricted to the two most confused classes
if len(off_diag) > 0:
    _, hard_r, hard_c = off_diag[0]
    mask = np.isin(label.cpu().numpy(), [hard_r, hard_c])
    hard_correct = confusion_matrix_accum[hard_r, hard_r] + confusion_matrix_accum[hard_c, hard_c]
    hard_total = (confusion_matrix_accum[hard_r, :].sum() +
                  confusion_matrix_accum[hard_c, :].sum())
    print(f"\nHard-pair accuracy (Class {hard_r} vs Class {hard_c}): "
          f"{hard_correct}/{hard_total} = {hard_correct / hard_total * 100:.2f}%")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title('Normalised Confusion Matrix (across all episodes)')
for r in range(args.way):
    for c in range(args.way):
        ax.text(c, r, f'{cm_norm[r, c]:.2f}', ha='center', va='center',
                color='white' if cm_norm[r, c] > 0.6 else 'black', fontsize=8)
plt.tight_layout()
cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# ================================================================== #
#  METRIC 2 — Distance Quality Analysis (Fisher Ratio)
# ================================================================== #
print("\n========== Metric 2: Distance Quality Analysis ==========")

intra = np.array(intra_distances)
inter = np.array(inter_distances)

mu_intra, std_intra = intra.mean(), intra.std()
mu_inter, std_inter = inter.mean(), inter.std()

# Fisher ratio: (mu_inter - mu_intra)^2 / (std_intra^2 + std_inter^2)
fisher_ratio = (mu_inter - mu_intra) ** 2 / (std_intra ** 2 + std_inter ** 2 + 1e-8)

print(f"Intra-class distances  — mean: {mu_intra:.4f}, std: {std_intra:.4f}")
print(f"Inter-class distances  — mean: {mu_inter:.4f}, std: {std_inter:.4f}")
print(f"Fisher Ratio (class-sep): {fisher_ratio:.4f}")

# KL divergence (inter || intra) as extra separability measure
# discretise to a shared histogram
all_d = np.concatenate([intra, inter])
bins = np.linspace(all_d.min(), all_d.max(), 60)
p_intra, _ = np.histogram(intra, bins=bins, density=True)
p_inter, _ = np.histogram(inter, bins=bins, density=True)
# smooth to avoid log(0)
p_intra = p_intra + 1e-8
p_inter = p_inter + 1e-8
p_intra /= p_intra.sum()
p_inter /= p_inter.sum()
kl_div = stats.entropy(p_inter, p_intra)
print(f"KL Divergence (inter||intra): {kl_div:.4f}")

# Plot distance distributions
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(intra, bins=50, alpha=0.6, color='steelblue', label='Intra-class', density=True)
ax.hist(inter, bins=50, alpha=0.6, color='tomato',    label='Inter-class', density=True)
ax.axvline(mu_intra, color='steelblue', linestyle='--', linewidth=1.5, label=f'μ_intra={mu_intra:.3f}')
ax.axvline(mu_inter, color='tomato',    linestyle='--', linewidth=1.5, label=f'μ_inter={mu_inter:.3f}')
ax.set_xlabel('Distance (−logit / temperature)')
ax.set_ylabel('Density')
ax.set_title(f'Distance Distribution  |  Fisher Ratio = {fisher_ratio:.3f}')
ax.legend()
plt.tight_layout()
dist_path = os.path.join(args.save_dir, 'distance_distribution.png')
plt.savefig(dist_path, dpi=150)
plt.close()
print(f"Distance distribution plot saved → {dist_path}")

# ================================================================== #
#  METRIC 3 — Complexity–Performance Trade-off
# ================================================================== #
print("\n========== Metric 3: Complexity–Performance Trade-off ==========")

latencies = np.array(latency_per_episode)
mean_lat   = latencies.mean()
std_lat    = latencies.std()
median_lat = np.median(latencies)

print(f"Inference latency per episode (meta/matching pass only):")
print(f"  Mean   : {mean_lat*1000:.2f} ms")
print(f"  Std    : {std_lat*1000:.2f} ms")
print(f"  Median : {median_lat*1000:.2f} ms")
print(f"  Min    : {latencies.min()*1000:.2f} ms")
print(f"  Max    : {latencies.max()*1000:.2f} ms")
print(f"Overall accuracy: {m:.4f} ± {pm:.4f} %")

# Plot accuracy vs latency (rolling window across episodes)
window = 50
rolled_acc = np.convolve(test_acc_record, np.ones(window)/window, mode='valid')
rolled_lat = np.convolve(latencies * 1000, np.ones(window)/window, mode='valid')  # ms

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: latency distribution over episodes
axes[0].plot(latencies * 1000, alpha=0.5, linewidth=0.8, color='darkorange')
axes[0].axhline(mean_lat * 1000, color='red', linestyle='--',
                label=f'Mean = {mean_lat*1000:.1f} ms')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Latency (ms)')
axes[0].set_title('Per-episode Inference Latency')
axes[0].legend()

# Right: accuracy vs latency scatter (rolling window)
axes[1].scatter(rolled_lat, rolled_acc, s=8, alpha=0.6, color='steelblue')
axes[1].set_xlabel('Avg Latency (ms, rolling window)')
axes[1].set_ylabel('Avg Accuracy (%)')
axes[1].set_title('Accuracy vs. Latency (Pareto view)')
# Annotate the single operating point of this model
axes[1].scatter([mean_lat * 1000], [m], s=120, color='red', zorder=5,
                label=f'This model\n({mean_lat*1000:.1f} ms, {m:.2f}%)')
axes[1].legend()

plt.suptitle('Complexity–Performance Trade-off', fontsize=13)
plt.tight_layout()
latency_path = os.path.join(args.save_dir, 'latency_accuracy.png')
plt.savefig(latency_path, dpi=150)
plt.close()
print(f"Latency–accuracy plot saved → {latency_path}")

# ================================================================== #
#  Summary
# ================================================================== #
print("\n========== Summary ==========")
print(f"Test Accuracy     : {m:.4f} ± {pm:.4f} %")
print(f"Fisher Ratio      : {fisher_ratio:.4f}  (higher = better class separation)")
print(f"KL Divergence     : {kl_div:.4f}  (higher = better separation)")
print(f"Mean Latency      : {mean_lat*1000:.2f} ms / episode")
print(f"Plots saved in    : {args.save_dir}/")