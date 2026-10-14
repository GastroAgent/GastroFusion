from pathlib import Path
import json

# ============================================================
# 根目录：只指定到这里
# ============================================================
ROOT = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data_tsy_827_doctor_test_split_test_test"
)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

SPLITS = ["train", "val", "test"]


def get_organ_name(folder_name):
    """
    十二指肠_1query -> 十二指肠
    食管_1query     -> 食管
    胃_1query       -> 胃
    """
    if folder_name.endswith("_1query"):
        return folder_name[:-len("_1query")]
    return folder_name


def build_label_map(organ_dir):
    """
    只根据 train/ 中的类别文件夹生成 label map
    """
    train_dir = organ_dir / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_dir}"
        )

    class_names = sorted([
        p.name
        for p in train_dir.iterdir()
        if p.is_dir()
    ])

    if not class_names:
        raise RuntimeError(
            f"No class folders found in: {train_dir}"
        )

    label_map = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    return label_map


def generate_split_txt(organ_dir, organ_name, split, label_map):
    """
    生成：
        十二指肠_train.txt
        十二指肠_val.txt
        十二指肠_test.txt

    每行形式：
        train/类别名/xxx.jpg 0
    """
    split_dir = organ_dir / split

    if not split_dir.exists():
        print(f"[SKIP] missing split: {split_dir}")
        return

    output_txt = organ_dir / f"{organ_name}_{split}.txt"

    # 检查 val/test 是否出现 train 中没有的新类别
    split_classes = {
        p.name
        for p in split_dir.iterdir()
        if p.is_dir()
    }

    unknown_classes = split_classes - set(label_map.keys())

    if unknown_classes:
        raise RuntimeError(
            f"{organ_name}/{split} contains unknown classes "
            f"not present in train: {sorted(unknown_classes)}"
        )

    lines = []
    total_images = 0

    print(f"\n[{organ_name}] {split}")

    for class_name, label in label_map.items():
        class_dir = split_dir / class_name

        # val/test 可以缺某个类别
        if not class_dir.exists():
            print(
                f"  [WARNING] missing class: "
                f"{class_name} (label={label})"
            )
            continue

        images = sorted([
            p
            for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ])

        print(
            f"  label={label:<2d} | "
            f"{class_name:<20s} | "
            f"n={len(images)}"
        )

        for image_path in images:
            relative_path = image_path.relative_to(organ_dir)

            lines.append(
                f"{relative_path.as_posix()} {label}"
            )

        total_images += len(images)

    with open(output_txt, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")

    print(f"  Saved: {output_txt}")
    print(f"  Total images: {total_images}")


def process_organ(organ_dir):
    organ_name = get_organ_name(organ_dir.name)

    print("\n" + "=" * 80)
    print(f"Processing organ: {organ_name}")
    print(f"Directory       : {organ_dir}")
    print("=" * 80)

    # ========================================================
    # 1. 根据 train 自动生成 label map
    # ========================================================
    label_map = build_label_map(organ_dir)

    print("\nLabel map:")
    for class_name, label in label_map.items():
        print(f"  {label}: {class_name}")

    # ========================================================
    # 2. 保存 label_map.json
    # ========================================================
    label_map_path = organ_dir / "label_map.json"

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            label_map,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nSaved label map: {label_map_path}")
    print(f"num_classes = {len(label_map)}")

    # ========================================================
    # 3. 生成 train / val / test txt
    # ========================================================
    for split in SPLITS:
        generate_split_txt(
            organ_dir=organ_dir,
            organ_name=organ_name,
            split=split,
            label_map=label_map
        )


def main():
    if not ROOT.exists():
        raise FileNotFoundError(
            f"Root directory not found: {ROOT}"
        )

    # ========================================================
    # 自动遍历所有 *_1query 目录
    # ========================================================
    organ_dirs = sorted([
        p
        for p in ROOT.iterdir()
        if p.is_dir()
        and p.name.endswith("_1query")
    ])

    if not organ_dirs:
        raise RuntimeError(
            f"No *_1query directories found under: {ROOT}"
        )

    print("=" * 80)
    print("Found organ directories:")
    for p in organ_dirs:
        print(f"  {p.name}")
    print("=" * 80)

    for organ_dir in organ_dirs:
        process_organ(organ_dir)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()