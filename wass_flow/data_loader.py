import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
import json
import numpy as np
import random
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

def paired_random_horizontal_flip(img1, img2, p=0.5):
    if random.random() < p:
        img1 = F.hflip(img1)
        img2 = F.hflip(img2)
    return img1, img2

def paired_random_vertical_flip(img1, img2, p=0.5):
    if random.random() < p:
        img1 = F.vflip(img1)
        img2 = F.vflip(img2)
    return img1, img2

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)  # List of image filenames
        self.transform = transform

        logger.info(f"Total number of images found: {len(self.image_filenames)}")

    def __len__(self):
        # Return the number of images
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load an image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        return image  # Return the image only (no labels)


class AddLocalGaussianNoise(object):
    def __init__(self, mean=0., std=0.05, probability=0.25, local_radius=36):
        self.mean = mean
        self.std = std
        self.probability = probability
        self.local_radius = local_radius  # 控制噪声应用的局部区域大小

    def __call__(self, img):
        if torch.rand(1) < self.probability:
            # 转换为numpy数组以便操作
            img_np = np.array(img)
            # 选择一个随机点作为噪声中心
            height, width = img_np.shape[:2]
            center_x = np.random.randint(self.local_radius, width - self.local_radius)
            center_y = np.random.randint(self.local_radius, height - self.local_radius)
            # 在选定中心周围生成一个高斯噪声块
            y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]
            mask = x * x + y * y <= self.local_radius * self.local_radius
            # 生成高斯噪声
            noise = np.zeros_like(img_np)
            noise[mask] = np.random.normal(self.mean, self.std, size=(mask.sum(), 3)) if len(
                img_np.shape) == 3 else np.random.normal(self.mean, self.std, size=(mask.sum()))
            # 应用噪声
            img_np[mask] += noise[mask]
            # 确保像素值仍在有效范围内
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            # 返回PIL图像
            return Image.fromarray(img_np)
        return img


base_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整尺寸
    transforms.RandomHorizontalFlip(0.25),  # 随机垂直翻转
    transforms.RandomVerticalFlip(0.25),  # 随机旋转-90到+90度
    # 温和的仿射变换
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=None, fill=0),
    transforms.ColorJitter(
        brightness=0.2,  # 亮度
        contrast=0.2,  # 对比度
        saturation=0.1,  # 饱和度
        hue=0.05  # 色调
    ),  # 颜色扰动，增强模型对颜色变化的鲁棒性
    transforms.RandomInvert(p=0.1),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))
    ], p=0.25),  # 有条件地应用高斯模糊，使用较小的kernel size以减少影响
    transforms.RandomApply([
        transforms.Grayscale(num_output_channels=3)
    ], p=0.25),  # 有一定概率转为灰度图
    AddLocalGaussianNoise(mean=0., std=0.3, probability=0.25, local_radius=36),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


class MedicalCLIPDiseaseDataset(Dataset):
    def __init__(self, data_path='/dev/shm/jmf/data/disease_data/filtered_食管疾病_5601.json', transform=transform,
                 base_transform=base_transform, label_path='/dev/shm/jmf/data/disease_data/diffusion_dit_labels.json',
                 text_key='caption', image_key='image', label_key='disease'):
        self.dataset = json.load(open(data_path, 'r'))
        self.dataset = [x for x in self.dataset if x['disease'] != '']
        self.diseases = {}
        labels = json.load(open(label_path, 'r'))
        self.labels, _ = zip(*labels)
        for x in self.dataset:
            disease = x[label_key]
            x.pop('domain')
            if disease not in self.diseases:
                self.diseases[disease] = [x]
            else:
                self.diseases[disease].append(x)

        self.transform = base_transform
        self.base_transform = base_transform
        self.text_key = text_key
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_file = data[self.image_key]
        image = Image.open(img_file).convert('RGB')
        if self.transform:
            src_img = self.transform(image)

        # tar_data = data
        tar_data = random.choice(self.diseases[data[self.label_key]])
        tar_img_file = tar_data[self.image_key]
        image = Image.open(tar_img_file).convert('RGB')

        if self.base_transform:
            tar_img = self.base_transform(image)
        else:
            tar_img = self.transform(image)

        # caption = tar_data[self.text_key]
        caption = tar_data['text'] + (('\n' + tar_data['disease']) if tar_data['disease'] != '' else '')
        label = tar_data[self.label_key]
        try:
            label_id = self.labels.index(label)
        except:
            label_id = 1000  # 1000 是无条件。
        if random.random() < 0.1:
            label_id = 1000  # 1000 是无条件。

        # tar_img = tar_img.permute(1, 2, 0)
        # src_img = src_img.permute(1, 2, 0)
        batch = dict(x1=tar_img, caption=caption, x0=src_img, class_id=torch.tensor(label_id).long())
        return batch


class MedicalCLIPAUGDataset(Dataset):
    def __init__(self, data_path='/home/lab/data/disease_data/diffusion_dit_dataset_tiny.json',
                 transform=base_transform, base_transform=base_transform,
                 label_path='/home/lab/data/disease_data/diffusion_dit_labels.json', text_key='caption',
                 image_key='image', label_key='disease'):
        self.dataset = json.load(open(data_path, 'r'))
        self.dataset = [x for x in self.dataset if x['disease'] != '']
        self.diseases = {}
        labels = json.load(open(label_path, 'r'))
        self.labels, _ = zip(*labels)
        for x in self.dataset:
            disease = x[label_key]
            x.pop('domain')
            if disease not in self.diseases:
                self.diseases[disease] = [x]
            else:
                self.diseases[disease].append(x)

        self.transform = transform
        self.base_transform = base_transform
        self.text_key = text_key
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_file = data[self.image_key]
        image = Image.open(img_file).convert('RGB')
        if self.transform:
            src_img = self.transform(image)
        # tar_data = data
        tar_data = random.choice(self.diseases[data[self.label_key]])
        tar_img_file = tar_data[self.image_key]
        image = Image.open(tar_img_file).convert('RGB')

        if self.base_transform:
            tar_img = self.base_transform(image)
        else:
            tar_img = self.transform(image)

        # caption = tar_data[self.text_key]
        caption = tar_data['text'] + (('\n' + tar_data['disease']) if tar_data['disease'] != '' else '')
        label = tar_data[self.label_key]
        try:
            label_id = self.labels.index(label)
        except:
            label_id = 999  # 1000 是无条件。
        # tar_img = tar_img.permute(1, 2, 0)
        # src_img = src_img.permute(1, 2, 0)
        batch = dict(x1=tar_img, caption=caption, x0=src_img, class_id=label_id)
        return batch

class MedicalA2BDataset(Dataset):
    def __init__(self,
                 data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_All_Cropped/A级',
                 data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_All_Cropped/C级',
                 label_A='食管炎-A级', label_B='食管炎-C级', y=2, shuffle_AB=False, caption=None,
                 transform=None, transform_A=None, transform_B=None):
        self.images_A = [os.path.join(data_path_A, x) for x in os.listdir(data_path_A)]
        self.images_B = [os.path.join(data_path_B, x) for x in os.listdir(data_path_B)]
        self.transform = transform
        self.label_A = label_A
        self.label_B = label_B
        self.y = y
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.shuffle_AB = shuffle_AB
        self.caption = caption
        self.hint_transform  = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.RandomGrayscale(p=1),  # 数据增强：20% 概率灰度化
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return min([len(self.images_A), len(self.images_B)])

    def __getitem__(self, idx):
        if len(self.images_A) == len(self.images_B) and not self.shuffle_AB:
            x0_path = self.images_A[idx]
            x1_path = self.images_B[idx]
        else:
            x0_path = random.choice(self.images_A)
            x1_path = random.choice(self.images_B)
        x0_path = x0_path
        x1_path = x1_path
        if self.transform_A is not None:
            x0 = self.transform_A(Image.open(x0_path).convert('RGB'))
        else:
            x0 = self.transform(Image.open(x0_path).convert('RGB'))

        if self.transform_B is not None:
            x1 = self.transform_B(Image.open(x1_path).convert('RGB'))
        else:
            x1 = self.transform(Image.open(x1_path).convert('RGB'))
        y = self.y
        if self.caption is None:
            # caption = f'将当前未知的医疗图像病症 转化为 {self.label_B} 病症的相关图像。'
            caption = f'{label_B}'
        else:
            caption = self.caption
            
        hint = self.hint_transform(Image.open(x1_path).convert('RGB'))
        
        
        batch = dict(x1=x1, caption=caption, x0=x0, label_A=self.label_A, label_B = self.label_B,
                     class_id=torch.tensor(y).long(),
                     x1_path=x1_path, x0_path=x0_path, hint = hint, hint_path=x1_path)
        return batch

class MedicalDataset(Dataset):
    def __init__(self,
                 image_dir='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_All_Cropped/A级',
                 label='食管炎A级', y=0, caption=None, transform=None):
        self.image_dir = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.transform = transform
        self.label = label
        self.y = y
        self.caption = caption

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        x_path = self.image_dir[idx]
        x = self.transform(Image.open(x_path).convert('RGB'))
        y = self.y
        if self.caption is None:
            caption = f'将当前未知的医疗图像病症 转化为 {self.label} 病症的相关图像。'
        else:
            caption = self.caption
        batch = dict(caption=caption, x=x,
                     class_id=torch.tensor(y).long(),
                     x_path=x_path)
        return batch

class MedicalIntraInterDataset(Dataset):
    """支持类内迁移(intra)与类间迁移(inter)的统一医学图像数据集。

    读取多个 JSON（每个 JSON 条目含 x0/x1/label_A/label_B/caption 等字段），
    构建按类组织的样本池：
      - x0_pool: {class: [{'path','label','label_id'}, ...]}   ← x0 只需 label
      - x1_pool: {class: [{'path','label','label_id','caption','hint_path'}, ...]}  ← x1 带 caption

    采样模式 (mode)：
      - 'intra' : 同类内采 x0/x1（内迁 c→c，配合 OT intra_class_suboptimal=True）
      - 'inter' : 不同类各采 x0/x1（互迁 c_i→c_j，标准最优 OT）
      - 'mixed' : 按 intra_prob 概率随机选 intra/inter

    返回的 batch 额外携带 ``is_intra`` 布尔标志，供训练循环据此设置 OT 模式。

    设计要点（与 MedicalJsonDataset 的区别）：
      - x0 只携带 label / label_id，不需要 caption；
      - caption 始终来自 x1（目标图），符合"caption 描述目标"的语义；
      - 内迁时从同一类池独立采 x0、x1，互迁时跨类组合。
    """

    def __init__(self,
                 json_glob=None,
                 json_paths=None,
                 mode='mixed',
                 intra_prob=0.15,
                 transform=None,
                 transform_A=None,
                 transform_B=None,
                 hint_transform=None,
                 flip=True,
                 pflip=0.25,
                 min_class_size=2,
                 **kwargs):
        """
        Args:
            json_glob: JSON 文件的 glob pattern（与 json_paths 二选一）。
            json_paths: JSON 路径列表（与 json_glob 二选一）。
            mode: 'intra' | 'inter' | 'mixed'。
            intra_prob: mode='mixed' 时内迁的概率。
            transform / transform_A / transform_B / hint_transform: 图像变换。
            flip / pflip: x1-hint 配对随机翻转。
            min_class_size: 类内池最小样本数，低于此值的类被跳过（内迁需 ≥2）。
        """
        super().__init__()
        assert mode in ('intra', 'inter', 'mixed'), f"Invalid mode: {mode}"
        self.mode = mode
        self.intra_prob = intra_prob
        self.transform = transform
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.hint_transform = hint_transform
        self.flip = flip
        self.pflip = pflip

        # ── 收集 JSON 路径 ──
        if json_paths is None:
            if json_glob is None:
                raise ValueError("Either json_glob or json_paths must be provided")
            import glob as _glob
            json_paths = sorted(_glob.glob(json_glob))
        elif isinstance(json_paths, str):
            json_paths = [json_paths]
        if len(json_paths) == 0:
            raise ValueError("No JSON files found")

        # ── 构建按类组织的样本池 ──
        self.x0_pool = {}   # class -> list of dict (path, label, label_id)
        self.x1_pool = {}   # class -> list of dict (path, caption, hint_path, label, label_id)
        _x0_dedup = {}      # class -> set(path) 用于 x0 去重
        _x1_dedup = {}      # class -> set((path, caption)) 用于 x1 去重

        for jp in tqdm(json_paths, desc='Loading JSON for IntraInterDataset'):
            try:
                data = json.load(open(jp, 'r'))
            except Exception as e:
                logger.warning(f"Failed to load {jp}: {e}")
                continue
            for e in data:
                try:
                    la = e['label_A']
                    lb = e['label_B']
                    la_id = e.get('label_A_id', -1)
                    lb_id = e.get('label_B_id', -1)
                    x0p = e['x0']
                    x1p = e['x1']
                    caption = e.get('caption', '')
                    hint_path = e.get('hint_path', x1p)
                except KeyError as ke:
                    logger.warning(f"Entry missing key {ke}, skipped: {e}")
                    continue

                # ── x0_pool：按 label_A 组织，path 去重 ──
                if la not in self.x0_pool:
                    self.x0_pool[la] = []
                    _x0_dedup[la] = set()
                if x0p not in _x0_dedup[la]:
                    _x0_dedup[la].add(x0p)
                    self.x0_pool[la].append({
                        'path': x0p, 'label': la, 'label_id': la_id,
                    })

                # ── x1_pool：按 label_B 组织，(path, caption) 去重 ──
                if lb not in self.x1_pool:
                    self.x1_pool[lb] = []
                    _x1_dedup[lb] = set()
                _x1_key = (x1p, caption)
                if _x1_key not in _x1_dedup[lb]:
                    _x1_dedup[lb].add(_x1_key)
                    self.x1_pool[lb].append({
                        'path': x1p, 'caption': caption, 'hint_path': hint_path,
                        'label': lb, 'label_id': lb_id,
                    })

        del _x0_dedup, _x1_dedup

        # ── 过滤小类 + 取交集（两类池都需有该类）──
        common_classes = set(self.x0_pool.keys()) & set(self.x1_pool.keys())
        self.classes = sorted([
            c for c in common_classes
            if len(self.x0_pool[c]) >= min_class_size
            and len(self.x1_pool[c]) >= min_class_size
        ])
        if len(self.classes) < 1:
            raise ValueError(
                f"No valid classes with >= {min_class_size} samples in both pools"
            )
        if len(self.classes) < 2 and mode in ('inter', 'mixed'):
            logger.warning(
                f"Only {len(self.classes)} valid class(es); "
                f"inter-class sampling will be impossible. Consider mode='intra'."
            )

        self._len = sum(len(self.x1_pool[c]) for c in self.classes)
        logger.info(
            f"[MedicalIntraInterDataset] classes={len(self.classes)} "
            f"mode={mode} intra_prob={intra_prob} "
            f"x1_total={self._len}"
        )

    def __len__(self):
        return self._len

    # ── 采样原语 ──
    def _sample_intra(self):
        """同类内采 (x0, x1)。"""
        c = random.choice(self.classes)
        s0 = random.choice(self.x0_pool[c])
        s1 = random.choice(self.x1_pool[c])
        return s0, s1

    def _sample_inter(self):
        """不同类各采一张 (x0, x1)。"""
        if len(self.classes) < 2:
            return self._sample_intra()  # fallback
        c_i, c_j = random.sample(self.classes, 2)
        s0 = random.choice(self.x0_pool[c_i])
        s1 = random.choice(self.x1_pool[c_j])
        return s0, s1

    def __getitem__(self, idx):
        # ── 决定内迁 / 互迁 ──
        if self.mode == 'intra':
            s0, s1 = self._sample_intra()
            is_intra = True
        elif self.mode == 'inter':
            s0, s1 = self._sample_inter()
            is_intra = False
        else:  # mixed
            if random.random() < self.intra_prob:
                s0, s1 = self._sample_intra()
                is_intra = True
            else:
                s0, s1 = self._sample_inter()
                is_intra = False

        # ── 加载 x0 ──
        x0_img = Image.open(s0['path']).convert('RGB')
        if self.transform_A is not None:
            x0 = self.transform_A(x0_img)
        else:
            x0 = self.transform(x0_img)

        # ── 加载 x1 + hint ──
        x1_image = Image.open(s1['path']).convert('RGB')
        hint_image = Image.open(s1['hint_path']).convert('RGB')
        if self.flip:
            x1_image, hint_image = paired_random_horizontal_flip(x1_image, hint_image, p=self.pflip)
            x1_image, hint_image = paired_random_vertical_flip(x1_image, hint_image, p=self.pflip)
        if self.transform_B is not None:
            x1 = self.transform_B(x1_image)
        else:
            x1 = self.transform(x1_image)
        if self.hint_transform is not None:
            hint = self.hint_transform(hint_image)
        else:
            hint = self.hint_transform(hint_image)

        caption = s1['caption']
        # ── 组装 batch（与 MedicalJsonDataset 对齐 + is_intra 标志）──
        batch = dict(
            caption=caption,
            x0=x0, x1=x1,
            label_A=s0['label'], label_B=s1['label'],
            label_A_id=s0['label_id'], label_B_id=s1['label_id'],
            class_id=torch.tensor(s1['label_id']).long(),  # 兼容现有训练脚本（目标类 id）
            x0_path=s0['path'], x1_path=s1['path'],
            hint=hint,
            is_intra=is_intra,
        )
        return batch


class MedicalDataPoolDataset(Dataset):
    """从 ``data/data_pool`` 目录加载的单图分类数据集，支持类内/类间采样。

    ``data_pool`` 中每个 JSON 文件名是一个类别名，内部记录只含**单张图**
    （字段 ``x`` / ``caption`` / ``hint_path`` / ``label`` / ``label_id``）。

    由于数据是单图（非配对），x0 和 x1 从同一个按类组织的池中采样：

      - ``intra``（类内）：同一类内独立采两张图作为 x0、x1（内迁 c→c）
      - ``inter``（类间）：不同类各采一张作为 x0、x1（互迁 c_i→c_j）
      - ``mixed``：按 ``intra_prob`` 概率随机选 intra / inter

    返回的 batch 字段与 :class:`MedicalIntraInterDataset` 完全一致（额外携带
    ``is_intra`` 标志），可无缝替换，训练脚本无需改动读取逻辑。
    """

    def __init__(self,
                 data_pool_dir='/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_pool',
                 mode='mixed',
                 intra_prob=0.15,
                 transform=None,
                 transform_A=None,
                 transform_B=None,
                 hint_transform=None,
                 flip=True,
                 pflip=0.25,
                 min_class_size=2,
                 **kwargs):
        """
        Args:
            data_pool_dir: ``data_pool`` 目录路径（内含多个 ``<类别>.json``）。
            mode: ``'intra'`` | ``'inter'`` | ``'mixed'``。
            intra_prob: ``mode='mixed'`` 时内迁的概率。
            transform / transform_A / transform_B / hint_transform: 图像变换。
            flip / pflip: x1-hint 配对随机翻转。
            min_class_size: 类内池最小样本数，低于此值的类被跳过（内迁需 ≥2）。
        """
        super().__init__()
        assert mode in ('intra', 'inter', 'mixed'), f"Invalid mode: {mode}"
        self.mode = mode
        self.intra_prob = intra_prob
        self.transform = transform
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.hint_transform = hint_transform
        self.flip = flip
        self.pflip = pflip

        # ── 收集 JSON 路径 ──
        import glob as _glob
        json_paths = sorted(_glob.glob(data_pool_dir))
        if len(json_paths) == 0:
            raise ValueError(f"No JSON files found in {data_pool_dir}")

        # ── 构建按类（记录内 label 字段）组织的样本池 ──
        # 单图数据：x0 和 x1 共用同一个池
        self.class_pool = {}   # label -> list of dict(path, caption, hint_path, label, label_id)

        for jp in tqdm(json_paths, desc='Loading data_pool JSON'):
            try:
                data = json.load(open(jp, 'r'))
            except Exception as e:
                logger.warning(f"Failed to load {jp}: {e}")
                continue
            for e in data:
                try:
                    path = e['x']
                    caption = e.get('caption', '')
                    hint_path = e.get('hint_path', path)
                    label = e.get('label', '')
                    label_id = e.get('label_id', -1)
                except KeyError as ke:
                    logger.warning(f"Entry missing key {ke}, skipped: {e}")
                    continue
                if label not in self.class_pool:
                    self.class_pool[label] = []
                self.class_pool[label].append({
                    'path': path,
                    'caption': caption,
                    'hint_path': hint_path,
                    'label': label,
                    'label_id': label_id,
                })

        # ── 过滤小类 ──
        self.classes = sorted([
            c for c, items in self.class_pool.items()
            if len(items) >= min_class_size
        ])
        if len(self.classes) < 1:
            raise ValueError(
                f"No valid classes with >= {min_class_size} samples"
            )
        if len(self.classes) < 2 and mode in ('inter', 'mixed'):
            logger.warning(
                f"Only {len(self.classes)} valid class(es); "
                f"inter-class sampling will be impossible. Consider mode='intra'."
            )

        self._len = sum(len(self.class_pool[c]) for c in self.classes)
        logger.info(
            f"[MedicalDataPoolDataset] classes={len(self.classes)} "
            f"mode={mode} intra_prob={intra_prob} "
            f"total={self._len}"
        )

    def __len__(self):
        return self._len

    # ── 采样原语 ──
    def _sample_intra(self):
        """同类内采两张不同的图作为 (x0, x1)。"""
        c = random.choice(self.classes)
        pool = self.class_pool[c]
        if len(pool) >= 2:
            s0, s1 = random.sample(pool, 2)
        else:
            # 退化情况：该类只有 1 张图，x0 和 x1 取同一张
            s0 = s1 = random.choice(pool)
        return s0, s1

    def _sample_inter(self):
        """不同类各采一张作为 (x0, x1)。"""
        if len(self.classes) < 2:
            return self._sample_intra()  # fallback
        c_i, c_j = random.sample(self.classes, 2)
        s0 = random.choice(self.class_pool[c_i])
        s1 = random.choice(self.class_pool[c_j])
        return s0, s1

    def __getitem__(self, idx):
        # ── 决定内迁 / 互迁 ──
        if self.mode == 'intra':
            s0, s1 = self._sample_intra()
            is_intra = True
        elif self.mode == 'inter':
            s0, s1 = self._sample_inter()
            is_intra = False
        else:  # mixed
            if random.random() < self.intra_prob:
                s0, s1 = self._sample_intra()
                is_intra = True
            else:
                s0, s1 = self._sample_inter()
                is_intra = False

        # ── 加载 x0 ──
        x0_img = Image.open(s0['path']).convert('RGB')
        if self.transform_A is not None:
            x0 = self.transform_A(x0_img)
        else:
            x0 = self.transform(x0_img)

        # ── 加载 x1 + hint ──
        x1_image = Image.open(s1['path']).convert('RGB')
        hint_image = Image.open(s1['hint_path']).convert('RGB')
        if self.flip:
            x1_image, hint_image = paired_random_horizontal_flip(x1_image, hint_image, p=self.pflip)
            x1_image, hint_image = paired_random_vertical_flip(x1_image, hint_image, p=self.pflip)
        if self.transform_B is not None:
            x1 = self.transform_B(x1_image)
        else:
            x1 = self.transform(x1_image)
        if self.hint_transform is not None:
            hint = self.hint_transform(hint_image)
        else:
            hint = self.hint_transform(hint_image)

        caption = s1['caption']
        # ── 组装 batch（与 MedicalIntraInterDataset 对齐 + is_intra 标志）──
        batch = dict(
            caption=caption,
            x0=x0, x1=x1,
            label_A=s0['label'], label_B=s1['label'],
            label_A_id=s0['label_id'], label_B_id=s1['label_id'],
            class_id=torch.tensor(s1['label_id']).long(),  # 兼容现有训练脚本（目标类 id）
            x0_path=s0['path'], x1_path=s1['path'],
            hint=hint,
            is_intra=is_intra,
        )
        assert is_intra == (s1['label'] == s0['label']), ""
        return batch


class MedicalJsonDataset(Dataset):
    def __init__(self,
                 path='',
                 transform=None,
                 transform_A=None,
                 transform_B=None,
                 hint_transform=None,
                 transform_mask=None,
                 flip=True,
                 pflip=0.25):
        dataset = json.load(open(path, 'r'))
        self.dataset = dataset      
        self.transform = transform
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.hint_transform = hint_transform
        self.transform_mask = transform_mask if transform_mask is None else self.transform
        self.flip = flip
        self.pflip = pflip
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        question_id = data['question_id']
        try:
            data_id = data['data_id']
        except:
            data_id = -1
        x0_path = data['x0']
        if self.transform_A is not None:
            x0 = self.transform_A(Image.open(x0_path).convert('RGB'))
        else:
            x0 = self.transform(Image.open(x0_path).convert('RGB'))

        x1_path = data['x1']
        x1_image = Image.open(x1_path).convert('RGB')
        hint_image = Image.open(data['hint_path']).convert('RGB')
        if self.flip:
            x1_image, hint_image = paired_random_horizontal_flip(x1_image, hint_image, p=self.pflip)
            x1_image, hint_image = paired_random_vertical_flip(x1_image, hint_image, p=self.pflip)
        if self.transform_B is not None:
            x1 = self.transform_B(x1_image)
        else:
            x1 = self.transform(x1_image)
        if self.hint_transform is not None:
            hint = self.hint_transform(hint_image)
        else:
            hint = self.hint_transform(hint_image)
        y = label_B_id = data['label_B_id']
        label_A_id = data['label_A_id']
        caption = data.get('caption', None)
        label_A = data['label_A']
        label_B = data['label_B']
        caption = f'{label_B}'
        batch = dict(caption=caption, x0=x0, x1=x1, label_A=label_A, label_B=label_B,
                     class_id=torch.tensor(y).long(), question_id=question_id,
                     x0_path=x0_path, x1_path=x1_path, data_id=data_id,
                     label_A_id=label_A_id, label_B_id=label_B_id,
                     hint=hint)
        
        try:                        
            if 'mask_hint_path' in data and data['mask_hint_path'] is not None:
                mask_hint_path = data['mask_hint_path']
                mask_hint = self.transform_mask(Image.open(mask_hint_path).convert('RGB'))
                batch.update({
                    'mask_hint_path': mask_hint_path,
                    'mask_hint': mask_hint
                })
        except:
            pass
        
        return batch

# class MedicalJsonDataset(Dataset):
#     def __init__(self,
#                  path='',
#                  transform=None,
#                  transform_A=None,
#                  transform_B=None,
#                  hint_transform=None,
#                  transform_mask=None):
#         dataset = json.load(open(path, 'r'))
#         self.dataset = dataset      
#         self.transform = transform
#         self.transform_A = transform_A
#         self.transform_B = transform_B
#         self.hint_transform = hint_transform
#         self.transform_mask = transform_mask if transform_mask is None else self.transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         data = self.dataset[idx]
#         question_id = data['question_id']
#         try:
#             data_id = data['data_id']
#         except:
#             data_id = -1
#         x0_path = data['x0']
#         if self.transform_A is not None:
#             x0 = self.transform_A(Image.open(x0_path).convert('RGB'))
#         else:
#             x0 = self.transform(Image.open(x0_path).convert('RGB'))

#         x1_path = data['x1']
#         x1_image = Image.open(x1_path).convert('RGB')
#         x1_image, _ = paired_random_horizontal_flip(x1_image, x1_image, p=0.25)
#         x1_image, _ = paired_random_vertical_flip(x1_image, x1_image, p=0.25)
#         if self.transform_B is not None:
#             x1 = self.transform_B(x1_image)
#         else:
#             x1 = self.transform(x1_image)
#         y = label_B_id = data['label_B_id']
#         label_A_id = data['label_A_id']
#         caption = data.get('caption', None)
#         label_A = data['label_A']
#         label_B = data['label_B']
#         caption = f'将当前未知的医疗病症图像 转化为 {label_B} 等相关病症图像。'
#         # caption = f'{label_B}'
#         batch = dict(caption=caption, x0=x0, x1=x1, label_A=label_A, label_B=label_B,
#                      class_id=torch.tensor(y).long(), question_id=question_id,
#                      x0_path=x0_path, x1_path=x1_path, data_id=data_id,
#                      label_A_id=label_A_id, label_B_id=label_B_id)
        
#         if 'hint_path' in data and data['hint_path'] is not None:
#             hint_path = data['hint_path']
#             if hint_path == x1_path and self.hint_transform is not None:
#                 hint = x1_image
#                 batch.update({
#                     'hint': self.hint_transform(hint)
#                 })
#             else:
#                 hint = Image.open(hint_path).convert('RGB')
#                 batch.update({
#                     'hint_path': hint_path,
#                     'hint': hint
#                 })
                        
#         # if 'mask_hint_path' in data and data['mask_hint_path'] is not None:
#         #     mask_hint_path = data['mask_hint_path']
#         #     mask_hint = self.transform_mask(Image.open(mask_hint_path).convert('RGB'))
#         #     batch.update({
#         #         'mask_hint_path': mask_hint_path,
#         #         'mask_hint': mask_hint
#         #     })
        
#         return batch

class MedicalClSJsonDataset(Dataset):
    def __init__(self,
                 path='',
                 transform=None,
                 transform_A=None,
                 transform_B=None,
                 hint_transform=None,
                 transform_mask=None):
        self.transform = transform
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.hint_transform = hint_transform
        self.transform_mask = transform_mask if transform_mask is None else self.transform
        if isinstance(path, list):
            dataset = []
            for pa in path:
                dataset += json.load(open(pa, 'r'))
        else:
            dataset = json.load(open(path, 'r'))
        self.dataset = dataset
        # for data in tqdm(dataset):
            # try:
            #     question_id = data['question_id']
            #     try:
            #         data_id = data['data_id']
            #     except:
            #         data_id = -1
                
            #     x0_path = data['x0']
            #     if self.transform_A is not None:
            #         x0 = self.transform_A(Image.open(x0_path).convert('RGB'))
            #     else:
            #         x0 = self.transform(Image.open(x0_path).convert('RGB'))

            #     x1_path = data['x1']
            #     x1_image = Image.open(x1_path).convert('RGB')
            #     x1_image, _ = paired_random_horizontal_flip(x1_image, x1_image, p=0.25)
            #     x1_image, _ = paired_random_vertical_flip(x1_image, x1_image, p=0.25)
            #     if self.transform_B is not None:
            #         x1 = self.transform_B(x1_image)
            #     else:
            #         x1 = self.transform(x1_image)
            #     y = label_B_id = data['label_B_id']
            #     label_A_id = data['label_A_id']
            #     caption = data.get('caption', None)
            #     label_A = data['label_A']
            #     label_B = data['label_B']
            #     caption = f'将当前未知的医疗病症图像 转化为 {label_B} 等相关病症图像。'
            #     batch = dict(caption=caption, x0=x0, x1=x1, label_A=label_A, label_B=label_B,
            #          class_id=torch.tensor(y).long(), question_id=question_id,
            #          x0_path=x0_path, x1_path=x1_path, data_id=data_id,
            #          label_A_id=label_A_id, label_B_id=label_B_id)
        
            #     if 'hint_path' in data and data['hint_path'] is not None:
            #         hint_path = data['hint_path']
            #         if hint_path == x1_path and self.hint_transform is not None:
            #             hint = x1_image
            #             batch.update({
            #                 'hint': self.hint_transform(hint)
            #             })
            #         else:
            #             hint = Image.open(hint_path).convert('RGB')
            #             batch.update({
            #                 'hint_path': hint_path,
            #                 'hint': hint
            #             })
                        
            #     if 'mask_hint_path' in data and data['mask_hint_path'] is not None:
            #         mask_hint_path = data['mask_hint_path']
            #         mask_hint = self.transform_mask(Image.open(mask_hint_path).convert('RGB'))
            #         batch.update({
            #             'mask_hint_path': mask_hint_path,
            #             'mask_hint': mask_hint
            #         })
        
            #     self.dataset.append(batch)
            # except Exception as e:
            #     print(e)
            #     continue


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]       
        if True:
            if True:
                question_id = data['question_id']
                try:
                    data_id = data['data_id']
                except:
                    data_id = -1
                
                x0_path = data['x0']
                if self.transform_A is not None:
                    x0 = self.transform_A(Image.open(x0_path).convert('RGB'))
                else:
                    x0 = self.transform(Image.open(x0_path).convert('RGB'))

                x1_path = data['x1']
                x1_image = Image.open(x1_path).convert('RGB')
                x1_image, _ = paired_random_horizontal_flip(x1_image, x1_image, p=0.25)
                x1_image, _ = paired_random_vertical_flip(x1_image, x1_image, p=0.25)
                if self.transform_B is not None:
                    x1 = self.transform_B(x1_image)
                else:
                    x1 = self.transform(x1_image)
                y = label_B_id = data['label_B_id']
                label_A_id = data['label_A_id']
                caption = data.get('caption', None)
                label_A = data['label_A']
                label_B = data['label_B']
                # caption = f'将当前未知的医疗病症图像 转化为 {label_B} 等相关病症图像。'
                caption = f'{label_B}'
                batch = dict(caption=caption, x0=x0, x1=x1, label_A=label_A, label_B=label_B,
                     class_id=torch.tensor(y).long(), question_id=question_id,
                     x0_path=x0_path, x1_path=x1_path, data_id=data_id,
                     label_A_id=label_A_id, label_B_id=label_B_id)
        
                if 'hint_path' in data and data['hint_path'] is not None:
                    hint_path = data['hint_path']
                    if hint_path == x1_path and self.hint_transform is not None:
                        hint = x1_image
                        batch.update({
                            'hint': self.hint_transform(hint)
                        })
                    else:
                        hint = Image.open(hint_path).convert('RGB')
                        batch.update({
                            'hint_path': hint_path,
                            'hint': hint
                        })
                        
                # if 'mask_hint_path' in data and data['mask_hint_path'] is not None:
                #     mask_hint_path = data['mask_hint_path']
                #     mask_hint = self.transform_mask(Image.open(mask_hint_path).convert('RGB'))
                #     batch.update({
                #         'mask_hint_path': mask_hint_path,
                #         'mask_hint': mask_hint
                #     })
        return batch

if __name__ == "__main__":

    # Example image transformations
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),   # Resize images to 256x256
        transforms.CenterCrop((128, 256)),  # Crop the center 256x256
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    # Initialize the dataset
    image_dir = '/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/anime_images'  # Specify your image directory
    dataset = MedicalCLIPTestDataset(image_dir=image_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Adjust the batch size as needed
        shuffle=True,  # Shuffle the data at the beginning of each epoch
        num_workers=4  # Number of subprocesses to use for data loading
    )

    for images in dataloader:
        logger.info(f"Shape of images: {images.shape}")
