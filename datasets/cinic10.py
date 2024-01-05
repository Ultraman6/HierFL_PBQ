import random
import sys

import requests
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import tarfile
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from datasets.cifar_mnist import create_shared_data_loaders, split_data

url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"

def download_dataset(url, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    local_filename = os.path.join(dataset_dir, url.split('/')[-1])

    # 请求头部用于获取文件大小
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))

    # 使用tqdm显示下载进度条
    with requests.get(url, stream=True) as r, open(local_filename, 'wb') as f, tqdm(
        unit='B',  # 单位为Byte
        unit_scale=True,  # 自动选择合适的单位
        unit_divisor=1024,  # 以1024为基数来计算单位
        total=file_size,  # 总大小
        file=sys.stdout,  # 输出到标准输出
        desc=local_filename  # 描述信息
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    # 如果文件是tar.gz格式，则解压
    if local_filename.endswith('.tar.gz'):
        with tarfile.open(local_filename, 'r:gz') as tar:
            tar.extractall(path=dataset_dir)



def pil_loader(path):
    # Open path as file to avoid ResourceWarning
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CINIC10(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # Set the appropriate folder based on training or test
        folder = "train" if train else "test"

        self.data = []
        self.targets = []

        # Load data and targets
        for class_name in os.listdir(os.path.join(root, folder)):
            class_path = os.path.join(root, folder, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append(img_path)
                self.targets.append(class_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]

        image = pil_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

def get_cinic10(dataset_dir, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
    # 定义CINIC-10的数据预处理
    if args.model == 'cnn_complex':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'resnet18' or 'resnet18_YWX':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("this nn for cifar10 not implemented")

    # 检查数据集是否存在
    dataset_root = os.path.join(dataset_dir, 'cinic10')
    if not (os.path.exists(dataset_root)):
        print("CINIC-10数据集不存在，正在下载...")
        download_dataset(url, dataset_root) # 调用下载函数

    # 加载CINIC-10数据集
    train = datasets.ImageFolder(os.path.join(dataset_root, 'CINIC-10/train'), transform=transform_train)
    test = datasets.ImageFolder(os.path.join(dataset_root, 'CINIC-10/test'), transform=transform_test)

    # 根据 args.share_niid 的值创建共享数据加载器
    if args.niid_share == 1:
        share_loaders = create_shared_data_loaders(train, args)
    else:
        share_loaders = [None] * args.num_edges

    test_set_size = len(test)
    subset_size = int(test_set_size * args.test_ratio)  # 例如，保留20%的数据
    # 生成随机索引来创建子集
    indices = list(range(test_set_size))
    subset_indices = random.sample(indices, subset_size)
    # 创建子集
    subset = Subset(test, subset_indices)
    # 使用子集创建 DataLoader
    v_test_loader = DataLoader(subset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    train_loaders = split_data(train, args, kwargs)

    test_loaders = []
    if args.test_on_all_samples == 1:
        # 将整个测试集分配给每个客户端
        for i in range(args.num_clients):
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=args.test_batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(test_loader)
    else:
        test_loaders = split_data(test, args, kwargs)

    return train_loaders, test_loaders, share_loaders, v_test_loader