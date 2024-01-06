import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset
import os
from datasets.shakespare.download import download_shakespeare_dataset
from datasets.shakespare.language_utils import word_to_indices, letter_to_index, VOCAB_SIZE



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
        batched_x = data_x[i: i + batch_size]
        batched_y = data_y[i: i + batch_size]    #一定要转long类型，不然优化器无法读入
        batched_x = torch.from_numpy(np.asarray(process_x(batched_x))).long()
        batched_y = torch.from_numpy(np.asarray(process_y(batched_y))).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_shakespeare(dataset_root, batch_size):
    train_path = dataset_root + "/train"
    test_path = dataset_root + "/test"
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
    # 初始化全局测试数据的 x 和 y
    merged_test_x = []
    merged_test_y = []
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

        # 将当前客户端的测试数据合并到全局测试数据中
        merged_test_x.extend(test_data[u]["x"])
        merged_test_y.extend(test_data[u]["y"])

        client_idx += 1
    client_num = client_idx
    output_dim = VOCAB_SIZE

    # 将合并后的全局测试数据转换为字典
    merged_test_data = {"x": merged_test_x, "y": merged_test_y}
    # test_data_g = [item for client_data in test_data for item in client_data]
    return train_data_local_dict, test_data_local_dict, train_data_global, merged_test_data, client_num, output_dim


def merge_data_loaders(batched_data_list, target_client_num, kwargs):

    original_client_num = len(batched_data_list)
    clients_per_merged_client = original_client_num // target_client_num

    merged_data_loaders = []
    for i in range(0, original_client_num, clients_per_merged_client):
        # 合并指定范围内的 DataLoader
        datasets_to_merge = [batched_data_list[j] for j in range(i, min(i + clients_per_merged_client, original_client_num))]
        merged_dataset = ConcatDataset(datasets_to_merge)
        # print(type(merged_dataset))
        merged_loader = DataLoader(merged_dataset, batch_size=None, shuffle=True, **kwargs)
        merged_data_loaders.append(merged_loader)

    return merged_data_loaders


def get_shakespare(dataset_dir, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
    # 检查数据集是否存在
    dataset_root = os.path.join(dataset_dir, 'shakespare')
    if not (os.path.exists(dataset_root)):
        print("shakespare数据集不存在，正在下载...")
        download_shakespeare_dataset(dataset_dir)

    (train_data_local_dict, test_data_local_dict, train_data_global,
     test_data_global, client_num, output_dim) = load_partition_data_shakespeare(args.dataset_root + '/shakespare',
                                                                                 args.train_batch_size)

    train_loaders = merge_data_loaders(train_data_local_dict, args.num_clients, kwargs)
    test_loaders = merge_data_loaders(test_data_local_dict, args.num_clients, kwargs)

    # 这里返回的是dataloader
    return train_loaders, test_loaders, batch_data(test_data_global, args.test_batch_size)
