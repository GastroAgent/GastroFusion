import os.path as osp
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = '/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/prototypical-network-pytorch/data/胃_1query/'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        # setname: 'train' / 'val' / 'test'
        # 数据目录结构: ROOT_PATH/{train,val,test}/{class_name}/*.jpg
        split_path = osp.join(ROOT_PATH, setname)
        self.wnids = sorted([d for d in os.listdir(split_path)
                             if osp.isdir(osp.join(split_path, d))])

        data = []
        label = []

        for lb, wnid in enumerate(self.wnids):
            folder = osp.join(split_path, wnid)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    data.append(osp.join(folder, fname))
                    label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
