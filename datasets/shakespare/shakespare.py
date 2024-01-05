import json

import numpy as np
import requests
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from tqdm import tqdm
import os

from datasets.shakespare.download import download_shakespeare_dataset
from datasets.shakespare.language_utils import word_to_indices, letter_to_index, VOCAB_SIZE

class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    return y_batch


def batch_data(data, batch_size):
    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x = torch.from_numpy(np.asarray(process_x(batched_x)))
        batched_y = torch.from_numpy(np.asarray(process_y(batched_y)))
        batch_data.append((batched_x, batched_y))
    return batch_data

def load_partition_data_shakespeare( batch_size):
    train_path = "~/shakespeare/train"
    test_path = "~/shakespeare/test"
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    client_num = client_idx
    output_dim = VOCAB_SIZE

    return train_data_global, test_data_global, client_num, output_dim

# Example usage
if __name__ == "__main__":
    dataset_dir = '~/shakespeare'  # Replace with your desired path
    # 检查数据集是否存在
    dataset_root = os.path.join(dataset_dir, 'shakespare')
    if not (os.path.exists(dataset_root)):
        print("shakespare数据集不存在，正在下载...")
        download_shakespeare_dataset(dataset_dir)

    train_data_global, test_data_globalload_partition_data_shakespeare(8)

# def get_shakespare(dataset_dir, args):
#     is_cuda = args.cuda
#     kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
#
#     # 检查数据集是否存在
#     dataset_root = os.path.join(dataset_dir, 'shakespare')
#     if not (os.path.exists(dataset_root)):
#         print("shakespare数据集不存在，正在下载...")
#         download_shakespeare_dataset(dataset_dir)
#
#     train_data_global, test_data_global = load_partition_data_shakespeare(args.batch_size)
#     train_dataset = ShakespeareDataset(train_data_global)
#     test_dataset = ShakespeareDataset(test_data_global)
#
#
#     test_set_size = len(test)
#     subset_size = int(test_set_size * args.test_ratio)  # 例如，保留20%的数据
#     # 生成随机索引来创建子集
#     indices = list(range(test_set_size))
#     subset_indices = random.sample(indices, subset_size)
#     # 创建子集
#     subset = Subset(test, subset_indices)
#     # 使用子集创建 DataLoader
#     v_test_loader = DataLoader(subset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
#
#     train_loaders = split_data(train, args, kwargs)
#
#     test_loaders = []
#     if args.test_on_all_samples == 1:
#         # 将整个测试集分配给每个客户端
#         for i in range(args.num_clients):
#             test_loader = torch.utils.data.DataLoader(
#                 test, batch_size=args.test_batch_size, shuffle=False, **kwargs
#             )
#             test_loaders.append(test_loader)
#     else:
#         test_loaders = split_data(test, args, kwargs)
#
#     return train_loaders, test_loaders, v_test_loader
