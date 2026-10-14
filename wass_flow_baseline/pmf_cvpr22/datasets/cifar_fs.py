import numpy as np
import torchvision.transforms as transforms

def dataset_setting(nSupport, img_size=32):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    mean = [x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]]
    std = [x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([
                                         #transforms.RandomCrop(32, padding=4),
                                         transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                         transforms.RandomHorizontalFlip(),
                                         #lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    valTransform = transforms.Compose([#lambda x: np.asarray(x),
                                       transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       normalize])
    inputW, inputH, nbCls = img_size, img_size, 7

    # trainDir = './data/胃_1query/train/'
    # valDir = './data/胃_1query/val/'
    # testDir = './data/胃_1query/test/'
    trainDir = '/mnt/inaisfs/data/home/tansy_criait/jmf/data_tsy_827_doctor_test_split/食管_1query/train/'
    valDir   = '/mnt/inaisfs/data/home/tansy_criait/jmf/data_tsy_827_doctor_test_split/食管_1query/val/'
    testDir  = '/mnt/inaisfs/data/home/tansy_criait/jmf/data_tsy_827_doctor_test_split/食管_1query/test/'


#     episodeJson = './data/cifar-fs/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
#             else './data/cifar-fs/val1000Episode_5_way_5_shot.json'
    episodeJson  = None #没有episodeJson
    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
