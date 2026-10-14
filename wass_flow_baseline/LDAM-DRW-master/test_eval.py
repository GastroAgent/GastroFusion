#!/usr/bin/env python
"""
独立测试脚本：加载 checkpoint，在测试集上评测
输出：Acc、Precision、Recall、F1-score、混淆边界分析、预测置信度分析

Usage:
    python test_eval.py --data-path /path/to/data --arch resnet18
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True,
                        help='数据集路径（如 /path/to/胃_1query）')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='checkpoint 路径，默认在 data-path 同级的 checkpoint 目录自动查找')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 自动推断 checkpoint 路径
    # ------------------------------------------------------------------
    if args.ckpt_path is None:
        # 训练脚本的 store_name 格式:
        #   {数据集名}_medical_{arch}_{loss}_{rule}_ep{ep}_bs{bs}_lr{无点lr}
        # 例如: 胃_medical_resnet50_LDAM_DRW_ep200_bs8_lr0001
        data_name = os.path.basename(os.path.normpath(args.data_path))  # e.g. "胃_1query"
        store_data_name = data_name.replace('_1query', '')               # e.g. "胃"
        # 实际 checkpoint 放在项目根目录的 checkpoint/ 下（而非 data/ 下）
        # 训练时用 resnet18，需要对应上
        auto_store_name = "{}_medical_{}_{}_{}_ep{}_bs{}_lr0001".format(
            store_data_name, args.arch, "LDAM", "DRW", 200, 8)
        project_root = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/LDAM-DRW-master'
        args.ckpt_path = os.path.join(project_root, 'checkpoint', auto_store_name, 'ckpt.best.pth.tar')

    if not os.path.exists(args.ckpt_path):
        print("Error: Checkpoint not found at {}".format(args.ckpt_path))
        print("Please run train script first or specify --ckpt-path manually.")
        return

    # ------------------------------------------------------------------
    # 确定数据集名称（用于 log 目录）
    # ------------------------------------------------------------------
    data_name = os.path.basename(os.path.normpath(args.data_path))
    store_data_name = data_name.replace('_1query', '')

    # ------------------------------------------------------------------
    # 动态导入数据集类（支持 medical_dataset 或普通 ImageFolder）
    # ------------------------------------------------------------------
    try:
        from medical_dataset import MedicalImageDataset
        has_medical_dataset = True
    except ImportError:
        has_medical_dataset = False

    # ------------------------------------------------------------------
    # 创建模型
    # ------------------------------------------------------------------
    test_folder = os.path.join(args.data_path, 'test')
    num_classes = len([d for d in os.listdir(test_folder)
                       if os.path.isdir(os.path.join(test_folder, d))])
    model = models.__dict__[args.arch](pretrained=False)
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # ------------------------------------------------------------------
    # 加载权重
    # ------------------------------------------------------------------
    ckpt = torch.load(args.ckpt_path, map_location='cuda:{}'.format(args.gpu), weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda(args.gpu)
    model.eval()

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if has_medical_dataset:
        test_dataset = MedicalImageDataset(os.path.join(args.data_path, 'test'), transform=transform)
    else:
        from torchvision.datasets import ImageFolder
        test_dataset = ImageFolder(os.path.join(args.data_path, 'test'), transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    all_logits_list, all_probs_list, all_preds, all_targets = [], [], [], []
    with torch.no_grad():
        for input, target in test_loader:
            input = input.cuda(args.gpu, non_blocking=True)
            output = model(input)
            probs = F.softmax(output, dim=1)
            confs, preds = probs.max(dim=1)
            all_logits_list.append(output.cpu())
            all_probs_list.append(confs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = torch.cat(all_probs_list, dim=0).numpy()
    all_logits_mat = torch.cat(all_logits_list, dim=0).numpy()

    # 获取类别名
    if has_medical_dataset:
        classes = test_dataset.classes
    else:
        classes = [c[1] for c in sorted(test_dataset.class_to_idx.items(), key=lambda x: x[1])]

    num_cls = len(classes)

    # ------------------------------------------------------------------
    # 指标一：Precision / Recall / F1-score
    # ------------------------------------------------------------------
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(num_cls)), zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro', zero_division=0)
    overall_acc = accuracy_score(all_targets, all_preds) * 100

    print('=' * 70)
    print('[指标一] Precision / Recall / F1-score')
    print('=' * 70)
    print("{:<30} {:>8} {:>8} {:>8}  {}".format('Class', 'Prec', 'Rec', 'F1', 'Support'))
    print('-' * 70)
    for i in range(num_cls):
        support = int((all_targets == i).sum())
        print("[{}] {:<28} {:>7.2f}% {:>7.2f}% {:>7.2f}%  {:>6}".format(
            i, classes[i], prec[i] * 100, rec[i] * 100, f1[i] * 100, support))
    print('-' * 70)
    print("{:<30} {:>7.2f}% {:>7.2f}% {:>7.2f}%".format(
        'Macro Average', macro_p * 100, macro_r * 100, macro_f1 * 100))
    print("{:<30} {:>7.2f}% {:>7.2f}% {:>7.2f}%".format(
        'Micro Average', micro_p * 100, micro_r * 100, micro_f1 * 100))
    print("{:<30} {:>7.2f}%".format('Overall Accuracy', overall_acc))

    # ------------------------------------------------------------------
    # 指标二：混淆边界分析
    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('[指标二] 混淆边界分析（最难分类的类别对）')
    print('=' * 70)

    cm = confusion_matrix(all_targets, all_preds)
    off_diag = []
    for r in range(num_cls):
        for c in range(num_cls):
            if r != c:
                off_diag.append((int(cm[r, c]), r, c))
    off_diag.sort(reverse=True)

    print('Top-5 误分类类别对（真实类 -> 预测类）:')
    for cnt, r, c in off_diag[:5]:
        total = int(cm[r, :].sum())
        print("  Class {} ({}) -> Class {} ({}): {} 次误分 ({:.1f}% of Class {})".format(
            r, classes[r], c, classes[c], cnt, cnt / total * 100, r))

    per_cls_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8) * 100
    print()
    print('每类准确率:')
    for i in range(num_cls):
        print("  [{}] {:<28} {:.2f}%".format(i, classes[i], per_cls_acc[i]))

    hardest_idx = int(np.argmin(per_cls_acc))
    easiest_idx = int(np.argmax(per_cls_acc))
    print()
    print("最难分类类别: [{}] {} ({:.2f}%)".format(
        hardest_idx, classes[hardest_idx], per_cls_acc[hardest_idx]))
    print("最易分类类别: [{}] {} ({:.2f}%)".format(
        easiest_idx, classes[easiest_idx], per_cls_acc[easiest_idx]))

    # ------------------------------------------------------------------
    # 指标三：预测置信度 / 不确定性分析
    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('[指标三] 预测置信度 / 不确定性分析')
    print('=' * 70)

    correct_mask = all_preds == all_targets
    wrong_mask = ~correct_mask
    conf_correct = all_probs[correct_mask] if correct_mask.sum() > 0 else np.array([])
    conf_wrong = all_probs[wrong_mask] if wrong_mask.sum() > 0 else np.array([])

    print('整体置信度统计:')
    print("  样本总数: {}".format(len(all_probs)))
    print("  正确预测数: {}  ({:.1f}%)".format(correct_mask.sum(), correct_mask.sum() / len(all_probs) * 100))
    print("  错误预测数: {}  ({:.1f}%)".format(wrong_mask.sum(), wrong_mask.sum() / len(all_probs) * 100))
    print()
    print("{:<20} {:>10} {:>12} {:>12}".format('指标', '整体', '正确预测', '错误预测'))
    print('-' * 56)
    print("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%".format(
        '置信度 均值', all_probs.mean() * 100,
        conf_correct.mean() * 100 if len(conf_correct) > 0 else 0,
        conf_wrong.mean() * 100 if len(conf_wrong) > 0 else 0))
    print("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%".format(
        '置信度 中位数', np.median(all_probs) * 100,
        np.median(conf_correct) * 100 if len(conf_correct) > 0 else 0,
        np.median(conf_wrong) * 100 if len(conf_wrong) > 0 else 0))
    print("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%".format(
        '置信度 标准差', all_probs.std() * 100,
        conf_correct.std() * 100 if len(conf_correct) > 0 else 0,
        conf_wrong.std() * 100 if len(conf_wrong) > 0 else 0))
    print("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%".format(
        '置信度 最小值', all_probs.min() * 100,
        conf_correct.min() * 100 if len(conf_correct) > 0 else 0,
        conf_wrong.min() * 100 if len(conf_wrong) > 0 else 0))
    print("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%".format(
        '置信度 最大值', all_probs.max() * 100,
        conf_correct.max() * 100 if len(conf_correct) > 0 else 0,
        conf_wrong.max() * 100 if len(conf_wrong) > 0 else 0))

    # 预测 Margin
    sorted_logits = np.sort(all_logits_mat, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    print()
    print('预测 Margin (Top1_logit - Top2_logit):')
    print("  均值  : {:.2f}".format(margin.mean()))
    print("  标准差: {:.2f}".format(margin.std()))

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(all_probs)
    for b in range(n_bins):
        in_bin = (all_probs > bin_boundaries[b]) & (all_probs <= bin_boundaries[b + 1])
        if in_bin.sum() > 0:
            avg_conf = all_probs[in_bin].mean()
            avg_acc = correct_mask[in_bin].mean()
            ece += abs(avg_conf - avg_acc) * in_bin.sum() / total
    print("ECE (Expected Calibration Error): {:.2f}%  (越低越好)".format(ece * 100))
    print('=' * 70)

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    log_dir = 'log_tsy_new_chekpoint_shiguan_7way/{}'.format(store_data_name)
    os.makedirs(log_dir, exist_ok=True)
    out_file = os.path.join(log_dir, 'test_results_full.txt')

    with open(out_file, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('Full Test Results\n')
        f.write('Checkpoint: {}\n'.format(args.ckpt_path))
        f.write('Checkpoint epoch: {}, Val Acc: {:.3f}%\n'.format(ckpt['epoch'], ckpt['best_acc1']))
        f.write('=' * 70 + '\n\n')

        f.write('[指标一] Precision / Recall / F1-score\n')
        f.write('-' * 70 + '\n')
        f.write("{:<30} {:>8} {:>8} {:>8}  {}\n".format('Class', 'Prec', 'Rec', 'F1', 'Support'))
        f.write('-' * 70 + '\n')
        for i in range(num_cls):
            support = int((all_targets == i).sum())
            f.write("[{}] {:<28} {:>7.2f}% {:>7.2f}% {:>7.2f}%  {:>6}\n".format(
                i, classes[i], prec[i] * 100, rec[i] * 100, f1[i] * 100, support))
        f.write('-' * 70 + '\n')
        f.write("{:<30} {:>7.2f}% {:>7.2f}% {:>7.2f}%\n".format(
            'Macro Average', macro_p * 100, macro_r * 100, macro_f1 * 100))
        f.write("{:<30} {:>7.2f}% {:>7.2f}% {:>7.2f}%\n".format(
            'Micro Average', micro_p * 100, micro_r * 100, micro_f1 * 100))
        f.write("{:<30} {:>7.2f}%\n\n".format('Overall Accuracy', overall_acc))

        f.write('[指标二] 混淆边界分析\n')
        f.write('-' * 70 + '\n')
        for cnt, r, c in off_diag[:5]:
            total_r = int(cm[r, :].sum())
            f.write("  Class {} ({}) -> Class {} ({}): {} 次误分 ({:.1f}% of Class {})\n".format(
                r, classes[r], c, classes[c], cnt, cnt / total_r * 100, r))
        f.write('\n每类准确率:\n')
        for i in range(num_cls):
            f.write("  [{}] {:<28} {:.2f}%\n".format(i, classes[i], per_cls_acc[i]))
        f.write('\n最难分类类别: [{}] {} ({:.2f}%)\n'.format(
            hardest_idx, classes[hardest_idx], per_cls_acc[hardest_idx]))
        f.write('最易分类类别: [{}] {} ({:.2f}%)\n\n'.format(
            easiest_idx, classes[easiest_idx], per_cls_acc[easiest_idx]))

        f.write('[指标三] 预测置信度 / 不确定性分析\n')
        f.write('-' * 70 + '\n')
        f.write("样本总数: {}, 正确: {}, 错误: {}\n\n".format(
            len(all_probs), correct_mask.sum(), wrong_mask.sum()))
        f.write("{:<20} {:>10} {:>12} {:>12}\n".format('指标', '整体', '正确预测', '错误预测'))
        f.write('-' * 56 + '\n')
        f.write("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%\n".format(
            '置信度 均值', all_probs.mean() * 100,
            conf_correct.mean() * 100 if len(conf_correct) > 0 else 0,
            conf_wrong.mean() * 100 if len(conf_wrong) > 0 else 0))
        f.write("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%\n".format(
            '置信度 中位数', np.median(all_probs) * 100,
            np.median(conf_correct) * 100 if len(conf_correct) > 0 else 0,
            np.median(conf_wrong) * 100 if len(conf_wrong) > 0 else 0))
        f.write("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%\n".format(
            '置信度 标准差', all_probs.std() * 100,
            conf_correct.std() * 100 if len(conf_correct) > 0 else 0,
            conf_wrong.std() * 100 if len(conf_wrong) > 0 else 0))
        f.write("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%\n".format(
            '置信度 最小值', all_probs.min() * 100,
            conf_correct.min() * 100 if len(conf_correct) > 0 else 0,
            conf_wrong.min() * 100 if len(conf_wrong) > 0 else 0))
        f.write("{:<20} {:>9.2f}% {:>11.2f}% {:>11.2f}%\n\n".format(
            '置信度 最大值', all_probs.max() * 100,
            conf_correct.max() * 100 if len(conf_correct) > 0 else 0,
            conf_wrong.max() * 100 if len(conf_wrong) > 0 else 0))
        f.write('预测 Margin (Top1_logit - Top2_logit):\n')
        f.write("  均值  : {:.2f}\n  标准差: {:.2f}\n".format(margin.mean(), margin.std()))
        f.write("ECE (Expected Calibration Error): {:.2f}%\n".format(ece * 100))

    print()
    print('Full results saved to: {}'.format(out_file))


if __name__ == '__main__':
    main()
