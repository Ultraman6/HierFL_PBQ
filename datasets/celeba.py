import os

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, Subset

def get_celeba(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.celeba()
    test = datasets.celeba(os.path.join(dataset_root, 'mnist'), train=False,
                          download=True, transform=transform)
    # note: is_shuffle here also is a flag for differentiating train and test
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
        test_loaders = split_data(test, args, kwargs, is_shuffle=False)


    test_set_size = len(test)
    subset_size = int(test_set_size * args.test_ratio)  # 例如，保留20%的数据
    # 生成随机索引来创建子集
    indices = list(range(test_set_size))
    subset_indices = random.sample(indices, subset_size)
    # 创建子集
    subset = Subset(test, subset_indices)
    # 使用子集创建 DataLoader
    v_test_loader = DataLoader(subset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_test_loader