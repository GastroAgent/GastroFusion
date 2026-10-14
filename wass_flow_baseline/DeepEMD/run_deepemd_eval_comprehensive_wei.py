"""
DeepEMD 评估脚本 - 综合评估版本
整合基础评估 + 混淆边界分析 + 距离质量分析 + 复杂度-性能权衡
"""

import argparse
import json
import os
import time
import numpy as np

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats

from Models.dataloader.samplers import CategoriesSampler
from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *

# ============ 自定义数据集：直接读取 DATA_DIR/class/... 结构 ============
class DirectDataset(Dataset):
    """直接从 DATA_DIR/class/xxx.jpg 结构加载，不依赖 train/val/test 子目录"""

    def __init__(self, data_dir, transform=None):
        self.data = []
        self.label = []
        self.data_dir = data_dir
        self.transform = transform

        label_list = sorted([d for d in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, d))])
        for idx, cls_name in enumerate(label_list):
            cls_folder = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.data.append(os.path.join(cls_folder, img_name))
                    self.label.append(idx)

        self.num_class = len(set(self.label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        from PIL import Image
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def get_transform(self):
        image_size = 84
        return transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

# ============ 配置 ============
DATA_DIR = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/data/data_true/数据_merge/南充/胃'
MODEL_DIR = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/cifar_fs/胃-5shot-8way/max_acc.pth'
OUTPUT_DIR = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/compare_methods/南充/胃'
SAVE_DIR = os.path.join(OUTPUT_DIR, 'metric_results')

parser = argparse.ArgumentParser()
parser.add_argument('-way', type=int, default=3)
parser.add_argument('-shot', type=int, default=5)
parser.add_argument('-query', type=int, default=1)
parser.add_argument('-dataset', type=str, default='cifar_fs')
parser.add_argument('-set', type=str, default='test')
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-metric', type=str, default='cosine')
parser.add_argument('-norm', type=str, default='center')
parser.add_argument('-deepemd', type=str, default='fcn')
parser.add_argument('-feature_pyramid', type=str, default=None)
parser.add_argument('-num_patch', type=int, default=9)
parser.add_argument('-patch_list', type=str, default='2,3')
parser.add_argument('-patch_ratio', type=float, default=2)
parser.add_argument('-solver', type=str, default='opencv')
parser.add_argument('-sfc_lr', type=float, default=100)
parser.add_argument('-sfc_wd', type=float, default=0)
parser.add_argument('-sfc_update_step', type=float, default=100)
parser.add_argument('-sfc_bs', type=int, default=4)
parser.add_argument('-test_episode', type=int, default=100)
parser.add_argument('-gpu', default='1,2,5,6')
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
parser.add_argument('-model_dir', type=str, default=MODEL_DIR)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-save_dir', type=str, default=SAVE_DIR)

args = parser.parse_args()
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

print("="*60)
pprint(vars(args))
print("="*60)

os.makedirs(args.save_dir, exist_ok=True)

set_seed(args.seed)
num_gpu = set_gpu(args)

# 加载模型
print("Loading model...")
model = DeepEMD(args)
model = load_model(model, args.model_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# 使用自定义 DirectDataset，直接从 DATA_DIR/class/... 加载，不走 test 子目录
test_set = DirectDataset(args.data_dir)
test_set.transform = test_set.get_transform()
args.num_class = test_set.num_class

# 自动获取类别名称（按字母顺序）
class_names = sorted([d for d in os.listdir(args.data_dir)
                     if os.path.isdir(os.path.join(args.data_dir, d))])
print(f"\nClasses ({len(class_names)}):")
for idx, cls_name in enumerate(class_names):
    print(f"  Class {idx}: {cls_name}")

# 测试集
sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, num_workers=4, pin_memory=True)
tqdm_gen = tqdm.tqdm(loader)

# 初始化
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)

# ------------------------------------------------------------------ #
#  Accumulators for three extra metrics
# ------------------------------------------------------------------ #

# (1) Confusion boundary analysis
confusion_matrix_accum = np.zeros((args.way, args.way), dtype=np.int64)

# (2) Distance quality analysis
intra_distances = []
inter_distances = []

# (3) Complexity-Performance trade-off
latency_per_episode = []

# 存储所有episode结果
all_episode_results = []

print("\n开始评估...\n")

with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot

        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]

        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)

        # timed meta / matching pass
        t0 = time.perf_counter()
        model.module.mode = 'meta'
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_per_episode.append(t1 - t0)

        # 计算准确率
        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc

        # 获取预测
        preds = logits.argmax(dim=1).cpu().numpy()
        true_labels = label.cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # ---- (1) Confusion matrix ----
        for t, p in zip(true_labels, preds):
            confusion_matrix_accum[t, p] += 1

        # ---- (2) Distance quality ----
        dist_matrix = (-logits / args.temperature).cpu().numpy()
        for q_idx in range(dist_matrix.shape[0]):
            true_cls = true_labels[q_idx]
            for cls in range(args.way):
                d = dist_matrix[q_idx, cls]
                if cls == true_cls:
                    intra_distances.append(d)
                else:
                    inter_distances.append(d)

        # 获取当前episode选择的类别顺序
        class_order = true_labels[:args.way]

        # 记录每个query的结果
        episode_records = []
        for q_idx in range(args.way * args.query):
            query_data_idx = test_set.data[args.way * args.shot + q_idx]

            support_paths = {}
            for cls_idx_in_ep, actual_class_idx in enumerate(class_order):
                cls_name = class_names[actual_class_idx]
                cls_support_paths = []
                for shot_idx in range(args.shot):
                    data_idx = cls_idx_in_ep * args.shot + shot_idx
                    cls_support_paths.append(test_set.data[data_idx])
                support_paths[cls_name] = cls_support_paths

            episode_records.append({
                'query_idx': int(q_idx),
                'query_path': query_data_idx,
                'support_paths': support_paths,
                'gt_class_idx': int(true_labels[q_idx]),
                'gt_class': class_names[true_labels[q_idx]],
                'pred_class_idx': int(preds[q_idx]),
                'pred_class': class_names[preds[q_idx]],
                'correct': bool(preds[q_idx] == true_labels[q_idx]),
                'class_probs': {class_names[j]: float(probs[q_idx, j]) for j in range(len(class_names))}
            })

        all_episode_results.append({
            'episode_id': i - 1,
            'episode_acc': float(acc),
            'correct': int(acc / 100 * len(episode_records)),
            'total': len(episode_records),
            'latency_ms': float((t1 - t0) * 1000),
            'records': episode_records
        })

        m, pm = compute_confidence_interval(test_acc_record[:i])
        tqdm_gen.set_description(f'batch {i}: This episode:{acc:.2f}%  average:{m:.4f}+{pm:.4f}')

# ================================================================== #
#  METRIC 1 — Confusion Boundary Analysis
# ================================================================== #
print("\n========== Metric 1: Confusion Boundary Analysis ==========")

cm_norm = confusion_matrix_accum.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
cm_norm = cm_norm / (row_sums + 1e-8)

off_diag = []
for r in range(args.way):
    for c in range(args.way):
        if r != c:
            off_diag.append((confusion_matrix_accum[r, c], r, c))
off_diag.sort(reverse=True)

print("Top-5 most confused class pairs (true -> predicted, count):")
confusion_pairs_data = []
for count, r, c in off_diag[:5]:
    true_cls_name = class_names[r] if r < len(class_names) else f"Class_{r}"
    pred_cls_name = class_names[c] if c < len(class_names) else f"Class_{c}"
    pct = count / row_sums[r, 0] * 100
    print(f"  {true_cls_name} -> {pred_cls_name}: {count} misclassifications ({pct:.1f}% of class {r})")
    confusion_pairs_data.append({
        'true_class_idx': int(r),
        'true_class_name': true_cls_name,
        'pred_class_idx': int(c),
        'pred_class_name': pred_cls_name,
        'count': int(count),
        'percentage': float(pct)
    })

per_class_acc = np.diag(cm_norm) * 100
print("\nPer-class accuracy (%):")
per_class_acc_data = {}
for cls in range(args.way):
    cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
    print(f"  Class {cls} ({cls_name}): {per_class_acc[cls]:.2f}%")
    per_class_acc_data[cls_name] = float(per_class_acc[cls])

# Hard-pair accuracy
if len(off_diag) > 0:
    _, hard_r, hard_c = off_diag[0]
    hard_correct = confusion_matrix_accum[hard_r, hard_r] + confusion_matrix_accum[hard_c, hard_c]
    hard_total = (confusion_matrix_accum[hard_r, :].sum() +
                  confusion_matrix_accum[hard_c, :].sum())
    hard_pair_acc = hard_correct / hard_total * 100
    print(f"\nHard-pair accuracy (Class {hard_r} vs Class {hard_c}): "
          f"{hard_correct}/{hard_total} = {hard_pair_acc:.2f}%")
else:
    hard_pair_acc = 0.0

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xlabel('Predicted class', fontsize=11)
ax.set_ylabel('True class', fontsize=11)
ax.set_title('Normalised Confusion Matrix (across all episodes)', fontsize=12)

tick_labels = [f"{i}\n{class_names[i][:6]}" if i < len(class_names) else str(i) for i in range(args.way)]
ax.set_xticks(range(args.way))
ax.set_yticks(range(args.way))
ax.set_xticklabels(tick_labels, fontsize=7)
ax.set_yticklabels(tick_labels, fontsize=7)

for r in range(args.way):
    for c in range(args.way):
        ax.text(c, r, f'{cm_norm[r, c]:.2f}', ha='center', va='center',
                color='white' if cm_norm[r, c] > 0.6 else 'black', fontsize=7)
plt.tight_layout()
cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved -> {cm_path}")

# ================================================================== #
#  METRIC 2 — Distance Quality Analysis (Fisher Ratio)
# ================================================================== #
print("\n========== Metric 2: Distance Quality Analysis ==========")

intra = np.array(intra_distances)
inter = np.array(inter_distances)

mu_intra, std_intra = intra.mean(), intra.std()
mu_inter, std_inter = inter.mean(), inter.std()

fisher_ratio = (mu_inter - mu_intra) ** 2 / (std_intra ** 2 + std_inter ** 2 + 1e-8)

print(f"Intra-class distances  -- mean: {mu_intra:.4f}, std: {std_intra:.4f}")
print(f"Inter-class distances -- mean: {mu_inter:.4f}, std: {std_inter:.4f}")
print(f"Fisher Ratio (class-sep): {fisher_ratio:.4f}")

# KL divergence
all_d = np.concatenate([intra, inter])
bins = np.linspace(all_d.min(), all_d.max(), 60)
p_intra, _ = np.histogram(intra, bins=bins, density=True)
p_inter, _ = np.histogram(inter, bins=bins, density=True)
p_intra = p_intra + 1e-8
p_inter = p_inter + 1e-8
p_intra /= p_intra.sum()
p_inter /= p_inter.sum()
kl_div = stats.entropy(p_inter, p_intra)
print(f"KL Divergence (inter||intra): {kl_div:.4f}")

# Plot distance distributions
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(intra, bins=50, alpha=0.6, color='steelblue', label='Intra-class', density=True)
ax.hist(inter, bins=50, alpha=0.6, color='tomato', label='Inter-class', density=True)
ax.axvline(mu_intra, color='steelblue', linestyle='--', linewidth=1.5, label=f'u_intra={mu_intra:.3f}')
ax.axvline(mu_inter, color='tomato', linestyle='--', linewidth=1.5, label=f'u_inter={mu_inter:.3f}')
ax.set_xlabel('Distance (-logit / temperature)')
ax.set_ylabel('Density')
ax.set_title(f'Distance Distribution  |  Fisher Ratio = {fisher_ratio:.3f}')
ax.legend()
plt.tight_layout()
dist_path = os.path.join(args.save_dir, 'distance_distribution.png')
plt.savefig(dist_path, dpi=150)
plt.close()
print(f"Distance distribution plot saved -> {dist_path}")

# ================================================================== #
#  METRIC 3 — Complexity-Performance Trade-off
# ================================================================== #
print("\n========== Metric 3: Complexity-Performance Trade-off ==========")

latencies = np.array(latency_per_episode)
mean_lat = latencies.mean()
std_lat = latencies.std()
median_lat = np.median(latencies)

print(f"Inference latency per episode (meta/matching pass only):")
print(f"  Mean   : {mean_lat*1000:.2f} ms")
print(f"  Std    : {std_lat*1000:.2f} ms")
print(f"  Median : {median_lat*1000:.2f} ms")
print(f"  Min    : {latencies.min()*1000:.2f} ms")
print(f"  Max    : {latencies.max()*1000:.2f} ms")

# Plot accuracy vs latency
window = 50
rolled_acc = np.convolve(test_acc_record, np.ones(window)/window, mode='valid')
rolled_lat = np.convolve(latencies * 1000, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(latencies * 1000, alpha=0.5, linewidth=0.8, color='darkorange')
axes[0].axhline(mean_lat * 1000, color='red', linestyle='--',
                label=f'Mean = {mean_lat*1000:.1f} ms')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Latency (ms)')
axes[0].set_title('Per-episode Inference Latency')
axes[0].legend()

axes[1].scatter(rolled_lat, rolled_acc, s=8, alpha=0.6, color='steelblue')
axes[1].set_xlabel('Avg Latency (ms, rolling window)')
axes[1].set_ylabel('Avg Accuracy (%)')
axes[1].set_title('Accuracy vs. Latency (Pareto view)')
axes[1].scatter([mean_lat * 1000], [m], s=120, color='red', zorder=5,
                label=f'This model\n({mean_lat*1000:.1f} ms, {m:.2f}%)')
axes[1].legend()

plt.suptitle('Complexity-Performance Trade-off', fontsize=13)
plt.tight_layout()
latency_path = os.path.join(args.save_dir, 'latency_accuracy.png')
plt.savefig(latency_path, dpi=150)
plt.close()
print(f"Latency-accuracy plot saved -> {latency_path}")

# ================================================================== #
#  计算 per-class 统计
# ================================================================== #
class_correct = {c: 0 for c in class_names}
class_total = {c: 0 for c in class_names}
for ep in all_episode_results:
    for rec in ep['records']:
        gt = rec['gt_class']
        class_total[gt] += 1
        if rec['correct']:
            class_correct[gt] += 1

per_class_stats = {}
for c in class_names:
    if class_total[c] > 0:
        per_class_stats[c] = {
            'correct': class_correct[c],
            'total': class_total[c],
            'accuracy': class_correct[c] / class_total[c]
        }

# ================================================================== #
#  保存综合结果
# ================================================================== #
m, pm = compute_confidence_interval(test_acc_record)
print("\n" + "="*60)
print(f"DeepEMD Test Acc: {m:.4f} +/- {pm:.4f}")
print("="*60)

result = {
    'summary': {
        'overall_acc': float(m),
        'ci95': float(pm),
        'total_episodes': len(all_episode_results),
        'way': args.way,
        'shot': args.shot,
        'query': args.query
    },
    'metric1_confusion_boundary': {
        'confusion_matrix': confusion_matrix_accum.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'top5_confused_pairs': confusion_pairs_data,
        'per_class_accuracy': per_class_acc_data,
        'hard_pair_accuracy': float(hard_pair_acc),
        'confusion_matrix_plot': cm_path
    },
    'metric2_distance_quality': {
        'intra_class_dist_mean': float(mu_intra),
        'intra_class_dist_std': float(std_intra),
        'inter_class_dist_mean': float(mu_inter),
        'inter_class_dist_std': float(std_inter),
        'fisher_ratio': float(fisher_ratio),
        'kl_divergence': float(kl_div),
        'distance_distribution_plot': dist_path
    },
    'metric3_complexity_performance': {
        'latency_mean_ms': float(mean_lat * 1000),
        'latency_std_ms': float(std_lat * 1000),
        'latency_median_ms': float(median_lat * 1000),
        'latency_min_ms': float(latencies.min() * 1000),
        'latency_max_ms': float(latencies.max() * 1000),
        'latency_per_episode': [float(l * 1000) for l in latency_per_episode],
        'latency_accuracy_plot': latency_path
    },
    'per_class_stats': per_class_stats,
    'episode_results': all_episode_results
}

output_path = os.path.join(OUTPUT_DIR, 'deepemd_comprehensive_results.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到: {output_path}")

# ================================================================== #
#  Summary
# ================================================================== #
print("\n========== Summary ==========")
print(f"Test Accuracy     : {m:.4f} +/- {pm:.4f} %")
print(f"Fisher Ratio      : {fisher_ratio:.4f}  (higher = better class separation)")
print(f"KL Divergence     : {kl_div:.4f}  (higher = better separation)")
print(f"Mean Latency      : {mean_lat*1000:.2f} ms / episode")
print(f"Plots saved in    : {args.save_dir}/")
