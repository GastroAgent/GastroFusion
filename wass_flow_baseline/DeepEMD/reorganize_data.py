import os
import shutil

# 源数据路径 - 胃
source_query = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/data/自贡/自贡_胃_topick_doctor_check/query'
source_support = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/data/自贡/自贡_胃_topick_doctor_check/support'

# 目标路径
target_base = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/data/自贡_胃'

# 创建目标目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_base, split), exist_ok=True)

# 获取query和support中的所有类别
query_classes = sorted([d for d in os.listdir(source_query) if os.path.isdir(os.path.join(source_query, d))])
support_classes = sorted([d for d in os.listdir(source_support) if os.path.isdir(os.path.join(source_support, d))])

# 类别名称映射 (support -> query 统一名称)
# 处理名称不一致的情况
name_mapping_support_to_query = {
    '胃窦溃疡（A1期）【见query】': '胃窦溃疡（A1期）'
}

print(f"Query classes: {query_classes}")
print(f"Support classes: {support_classes}")

# 使用query的类别作为标准类别
all_classes = query_classes

# 将query和support的图片复制到test目录下对应的类别文件夹
for cls in all_classes:
    # 创建test下的类别文件夹
    test_cls_dir = os.path.join(target_base, 'test', cls)
    os.makedirs(test_cls_dir, exist_ok=True)

    # 复制query图片，添加前缀区分
    query_cls_dir = os.path.join(source_query, cls)
    if os.path.exists(query_cls_dir):
        query_images = os.listdir(query_cls_dir)
        for img in query_images:
            src = os.path.join(query_cls_dir, img)
            dst = os.path.join(test_cls_dir, f"query_{img}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"  {cls}: 复制 {len(query_images)} 张 query 图片")

    # 复制support图片，需要映射名称
    support_cls_name = name_mapping_support_to_query.get(cls, cls)
    support_cls_dir = os.path.join(source_support, support_cls_name)
    if os.path.exists(support_cls_dir):
        support_images = os.listdir(support_cls_dir)
        for img in support_images:
            src = os.path.join(support_cls_dir, img)
            dst = os.path.join(test_cls_dir, f"support_{img}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"  {cls}: 复制 {len(support_images)} 张 support 图片")

    # 统计该类别的总图片数
    total_images = len(os.listdir(test_cls_dir))
    print(f"  -> {cls} 共有 {total_images} 张图片")

# 同时为train和val创建符号链接，指向test（或者也可以为空，让代码只使用test）
# 这里选择复制一份到train和val，确保数据集完整
for split in ['train', 'val']:
    for cls in all_classes:
        split_cls_dir = os.path.join(target_base, split, cls)
        if not os.path.exists(split_cls_dir):
            # 创建符号链接指向test
            os.symlink(os.path.join(target_base, 'test', cls), split_cls_dir)

print(f"\n数据重组完成！目标路径: {target_base}")

# 验证结果
print("\n验证 test 目录结构:")
for cls in os.listdir(os.path.join(target_base, 'test')):
    cls_path = os.path.join(target_base, 'test', cls)
    print(f"  {cls}: {len(os.listdir(cls_path))} 张图片")
