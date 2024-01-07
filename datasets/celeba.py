import os
import random

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from datasets.cifar_mnist import split_data


def get_celeba(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load CelebA dataset
    train = datasets.CelebA(root=dataset_root, split='train', target_type='attr', transform=transform, download=True)
    test = datasets.CelebA(root=dataset_root, split='test', target_type='attr', transform=transform, download=True)

    # Custom function to split data (implementation depends on your specific needs)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)

    test_loaders = []
    if args.test_on_all_samples == 1:
        # 将整个测试集分配给每个客户端
        for i in range(args.num_clients):
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=args.test_batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(test_loader)
    else:
        # 平均分配测试集
        num_samples_per_client = len(test) // args.num_clients
        test_indices = list(range(len(test)))
        for i in range(args.num_clients):
            # 为每个客户端分配样本
            start_idx = i * num_samples_per_client
            end_idx = len(test) if i == args.num_clients - 1 else (i + 1) * num_samples_per_client
            client_indices = test_indices[start_idx:end_idx]
            # 创建每个客户端的数据加载器
            client_test_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(test, client_indices),
                batch_size=args.test_batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(client_test_loader)

    # Creating a validation subset from the test set
    test_set_size = len(test)
    subset_size = int(test_set_size * args.test_ratio)  # For example, retain 20% of the data
    indices = list(range(test_set_size))
    subset_indices = random.sample(indices, subset_size)
    subset = Subset(test, subset_indices)
    v_test_loader = DataLoader(subset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_test_loader