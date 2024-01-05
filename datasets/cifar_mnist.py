"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import json
import os
import random

from tqdm import tqdm
from sklearn import metrics
import numpy as np
import h5py
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn

import datasets.get_data

cudnn.banchmark = True
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, Subset
from options import args_parser


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target


def gen_ran_sum(_sum, num_users):
    base = 100 * np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100 * num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users


def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def iid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    可自定义每个客户端的训练样本量
    """
    # 数据装载初始化
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # if num_samples_per_client == -1, then use all samples
    if args.self_sample == -1:
        num_samples_per_client = int(len(dataset) / args.num_clients)
        for i in range(args.num_clients):
            dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
            data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                         batch_size=args.train_batch_size,
                                         shuffle=is_shuffle, **kwargs)
    else:  # 自定义每客户样本量开启
        # 提取映射关系参数并将其解析为JSON对象
        sample_mapping_json = args.sample_mapping
        sample_mapping = json.loads(sample_mapping_json)
        for i in range(args.num_clients):
            # 客户按id分配样本量
            sample = sample_mapping[str(i)]
            if sample == -1: sample = int(len(dataset) / args.num_clients)
            dict_users[i] = np.random.choice(all_idxs, sample, replace=False)
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
            data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                         batch_size=args.train_batch_size,
                                         shuffle=is_shuffle, **kwargs)

    return data_loaders


def iid_nesize_split(dataset, args, kwargs, is_shuffle=True):
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace=False)
        # dict_users[i] = dict_users[i].astype(int)
        # dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.train_batch_size,
                                     shuffle=is_shuffle, **kwargs)

    return data_loaders


def niid_esize_split(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    # each client has only two classes of the network
    num_shards = 2 * args.num_clients
    # the number of images in one shard
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test

    if args.dataset != "femnist":
        # original
        # editer: Ultraman6 20230928
        # torch>=1.4.0
        labels = dataset.targets
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # sort the data according to their label
        idxs = idxs_labels[0, :]
        idxs = idxs.astype(int)
    else:
        # custom
        labels = np.array(dataset.targets)  # 将labels转换为NumPy数组
        idxs_labels = np.vstack((idxs[:len(labels)], labels[:len(idxs)]))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        idxs = idxs.astype(int)

    # divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.train_batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def niid_esize_split_train(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    #     and record the split patter
    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.train_batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs)
    return data_loaders, split_pattern


def niid_esize_split_test(dataset, args, kwargs, split_pattern, is_shuffle=False):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.train_batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs)
    return data_loaders, None


def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        # split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
            # store the label
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.train_batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs)
    return data_loaders, split_pattern


def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle=False):
    """
    :param dataset: test dataset
    :param args:
    :param kwargs:
    :param split_pattern: split pattern from trainloaders
    :param test_size: length of testloader of each client
    :param is_shuffle: False for testloader
    :return:
    """
    data_loaders = [0] * args.num_clients
    # for mnist and cifar 10, only 10 classes
    num_shards = 10
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i]
        # idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.train_batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs)
    return data_loaders, None


def partition_completed(dataset, args, kwargs, is_shuffle=True):
    global proportions, net_dataidx_map, idx_batch
    np.random.seed(args.seed)
    random.seed(args.seed)

    n_train = len(dataset)
    y_train = np.array(dataset.targets)

    if args.partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, args.num_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_clients)}

    elif args.partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10

        # N = dataset.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.num_clients)]
            for k in range(args.num_class):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.beta_new, args.num_clients))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < n_train / args.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break
        for j in range(args.num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif args.partition > "noniid-#label0" and args.partition <= "noniid-#label9":
        num = eval(args.partition[13:])
        times=[0 for _ in range(args.num_class)]
        contain=[]
        for i in range(args.num_clients):
            current=[i % args.num_class]
            times[i % args.num_class]+=1
            j=1
            while (j<num):
                ind=random.randint(0, args.num_class - 1)
                if ind not in current:
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(args.num_clients)}
        for i in range(args.num_class):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(args.num_clients):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
        for i in range(args.num_clients):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()

    elif args.partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(args.beta_new, args.num_clients))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_clients)}
        for i in range(args.num_clients):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()

    data_loaders = []
    for client_id in net_dataidx_map:
        client_dataset = DatasetSplit(dataset, net_dataidx_map[client_id])
        client_loader = DataLoader(client_dataset, batch_size=args.train_batch_size, shuffle=is_shuffle, **kwargs)
        data_loaders.append(client_loader)

    return data_loaders

def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle=True):
    num_clients = args.num_clients
    num_shards = args.num_shards
    labels = np.array(dataset.targets)
    total_data = len(labels)

    # 根据标签排序数据集
    idxs = np.arange(total_data)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 将数据集分割成多个分片
    shard_size = total_data // num_shards
    shards = [idxs[i:i + shard_size] for i in range(0, total_data, shard_size)]

    # 确保每个客户端获得单一标签的数据
    dict_users = {i: np.array([]) for i in range(num_clients)}
    for client_id in range(num_clients):
        selected_shards = np.random.choice(range(num_shards), num_shards // num_clients, replace=False)
        for shard in selected_shards:
            dict_users[client_id] = np.concatenate((dict_users[client_id], shards[shard]), axis=0)

    # 创建数据加载器
    data_loaders = []
    for client_id in range(num_clients):
        client_dataset = DatasetSplit(dataset, dict_users[client_id])
        client_loader = DataLoader(client_dataset, batch_size=args.train_batch_size, shuffle=is_shuffle, **kwargs)
        data_loaders.append(client_loader)

    return data_loaders


# 如何调整本地训练样本数量
def split_data(dataset, args, kwargs, is_shuffle=True):
    """
    return dataloaders
    """
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:
        data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -2:
        data_loaders = niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle)
    elif args.iid == -3: # 开启新划分方法
        data_loaders = partition_completed(dataset, args, kwargs, is_shuffle)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders



def create_shared_data_loaders(train, args, **kwargs):
    """
    创建每个边缘服务器的共享数据加载器。
    :param train: 完整的训练数据集。
    :param args: 包含配置参数，如边缘服务器数量（num_edges）、每个客户的batchsize
    :return: 每个边缘服务器的共享数据加载器列表。
    """
    total_data_size = len(train)
    data_per_edge = int(0.05 * total_data_size)  # 每个边缘服务器分配 5% 的数据
    # 创建一个随机索引
    indices = np.arange(total_data_size)
    np.random.shuffle(indices)
    # 分配数据给每个边缘服务器
    edge_shared_data_loaders = []
    for eid in range(args.num_edges):
        # 计算为每个边缘服务器分配的数据的索引
        start_idx = eid * data_per_edge
        end_idx = min((eid + 1) * data_per_edge, total_data_size)
        # 创建数据子集
        subset_indices = indices[start_idx:end_idx]
        subset_data = Subset(train, subset_indices)
        # 创建 DataLoader
        shared_data_loader = DataLoader(subset_data, batch_size=args.train_batch_size, shuffle=True, **kwargs)
        # 添加到列表
        edge_shared_data_loaders.append(shared_data_loader)

    return edge_shared_data_loaders


def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                           download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
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


def get_cifar10(dataset_root, args):  # cifa10数据集下只能使用cnn_complex和resnet18模型
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
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
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True,
                             download=True, transform=transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False,
                            download=True, transform=transform_test)

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

    return train_loaders, test_loaders, v_test_loader


def get_femnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    train_h5 = h5py.File(os.path.join(dataset_root, 'femnist/fed_emnist_train.h5'), "r")
    test_h5 = h5py.File(os.path.join(dataset_root, 'femnist/fed_emnist_test.h5'), "r")
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    _EXAMPLE = "examples"
    _IMGAE = "pixels"
    _LABEL = "label"

    client_ids_train = list(train_h5[_EXAMPLE].keys())
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    train_ids = client_ids_train
    test_ids = client_ids_test

    for client_id in train_ids:
        train_x.append(train_h5[_EXAMPLE][client_id][_IMGAE][()])
        train_y.append(train_h5[_EXAMPLE][client_id][_LABEL][()].squeeze())
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    for client_id in test_ids:
        test_x.append(test_h5[_EXAMPLE][client_id][_IMGAE][()])
        test_y.append(test_h5[_EXAMPLE][client_id][_LABEL][()].squeeze())
    test_x = np.vstack(test_x)
    test_y = np.hstack(test_y)

    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)


    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_ds.targets = train_y  # 添加targets属性
    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_ds.targets = test_y  # 添加targets属性

    v_test_loader = DataLoader(test_ds, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    train_loaders = split_data(train_ds, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test_ds, args, kwargs, is_shuffle=False)

    train_h5.close()
    test_h5.close()

    return train_loaders, test_loaders, v_test_loader


def show_distribution(dataloader, args):
    """
    Show the distribution of the data on a certain client with dataloader.
    Return:
        Percentage of each class of the label.
    """
    # Retrieve labels
    if args.dataset in ['femnist', 'cifar10', 'mnist', 'synthetic', 'cinic10']:
        labels = [label for _, label in dataloader.dataset]
    elif args.dataset == 'fsdd':
        labels = dataloader.dataset.labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))

    # Ensure labels are numpy array and integer type
    labels = np.array(labels).astype(int)
    num_samples = len(labels)
    max_label = labels.max()

    # Initialize distribution array
    distribution = np.zeros(max_label + 1)

    # Calculate distribution
    for label in labels:
        distribution[label] += 1

    # Normalize to get percentages
    distribution = distribution / num_samples
    return distribution


if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = datasets.get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)
