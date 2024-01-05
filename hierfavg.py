# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)
import json
import time
from threading import Thread

from matplotlib import pyplot as plt
from torch import multiprocessing

from average import models_are_equal
from models.synthetic_logistic import LogisticRegression_SYNTHETIC
from options import args_parser
from tensorboardX import SummaryWriter
import torch
from client import Client
from edge import Edge
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
import copy
import numpy as np
from tqdm import tqdm
from models.mnist_cnn import mnist_lenet, mnist_cnn
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18, ResNet18_YWX
from models.mnist_logistic import LogisticRegression_MNIST
import os
import torch.multiprocessing as mp


def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[], [], [], [], [], [], [], [], [], []]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis


def get_edge_class(args, edges, clients):
    edge_class = [[], [], [], [], []]
    for (i, edge) in enumerate(edges):
        for cid in edge.cids:
            client = clients[cid]
            train_loader = client.train_loader
            distribution = show_distribution(train_loader, args)
            label = np.argmax(distribution)
            edge_class[i].append(label)
    print(f'class distribution among edge {edge_class}')


def initialize_edges_iid(num_edges, clients, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 10 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    # only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
            #     0-9 labels in total
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        edges.append(Edge(id=eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                          for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    # And the last one, eid == num_edges -1
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                      for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients


def initialize_edges_niid(num_edges, clients, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 5 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    # only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    label_ranges = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [6, 7, 8, 9, 0]]
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        label_range = label_ranges[eid]
        for i in range(2):
            for label in label_range:
                #     5 labels in total
                if len(client_class_dis[label]) > 0:
                    assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
                    client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
                else:
                    label_backup = 2
                    assigned_client_idx = np.random.choice(client_class_dis[label_backup], 1, replace=False)
                    client_class_dis[label_backup] = list(
                        set(client_class_dis[label_backup]) - set(assigned_client_idx))
                for idx in assigned_client_idx:
                    assigned_clients_idxes.append(idx)
        edges.append(Edge(id=eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                          for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()

    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                      for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients


def all_clients_test(server, clients, cids, device):
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge


def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all


def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            global_nn = LogisticRegression_MNIST(input_dim=1, output_dim=10)
        elif args.model == 'cnn':
            global_nn = mnist_cnn(input_channels=1, output_channels=10)
        else:
            raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'femnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=62)
        elif args.model == 'logistic':
            global_nn = LogisticRegression_MNIST(input_dim=1, output_dim=62)
        elif args.model == 'cnn':
            global_nn = mnist_cnn(input_channels=1, output_channels=62)
        else:
            raise ValueError(f"Model{args.model} not implemented for femnist")
    elif args.dataset == 'cifar10' or 'cinic10':
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            global_nn = ResNet18()
        elif args.model == 'resnet18_YWX':
            global_nn = ResNet18_YWX()
        else:
            raise ValueError(f"Model{args.model} not implemented for cifar")
    elif args.dataset == 'synthetic':
        if args.model == 'logistic':
            global_nn = LogisticRegression_SYNTHETIC(args.dimension, args.num_class)
    else:
        raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn


# 分层联邦总聚合
def HierFAVG(args):
    # make experiments repeatable
    global avg_acc_v
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    # Build dataloaders
    train_loaders, test_loaders, share_loaders, v_test_loader = get_dataloaders(args)
    if args.show_dis:
        # 显示客户端的训练集和测试集分布
        for i in range(args.num_clients):
            # 显示训练集分布
            train_loader = train_loaders[i]
            print("Train dataloader {} size: {}".format(i, len(train_loader.dataset)))
            train_distribution = show_distribution(train_loader, args)
            print("Train dataloader {} distribution:".format(i))
            print(train_distribution)
            # 显示测试集分布
            test_loader = test_loaders[i]
            test_size = len(test_loader.dataset)
            print("Test dataloader {} size: {}".format(i, test_size))
            if args.test_on_all_samples != 1:
                test_distribution = show_distribution(test_loader, args)
                print("Test dataloader {} distribution:".format(i))
                print(test_distribution)
        # 显示边缘服务器的共享数据分布
        if args.niid_share:
            for eid in range(args.num_edges):
                print("Edge {} shared data size: {}".format(eid, len(share_loaders[eid].dataset)))
                shared_data_distribution = show_distribution(share_loaders[eid], args)
                print("Edge {} shared data distribution:".format(eid))
                print(shared_data_distribution)
        else:
            print("Edges  has no shared data")
        print("Cloud valid data size: {}".format(len(v_test_loader.dataset)))
        valid_data_distribution = show_distribution(v_test_loader, args)
        print("Cloud valid data distribution: {}".format(valid_data_distribution))

    # 读取mapping配置信息
    mapping = None
    if args.active_mapping == 1:
        # 提取映射关系参数并将其解析为JSON对象
        mapping = json.loads(args.mapping)

    # 读取attack_mapping配置信息
    attack_mapping = None
    if args.attack_flag == 1: # 表示开启模型攻击
        # 提取映射关系参数并将其解析为JSON对象
        attack_mapping = json.loads(args.attack_mapping)


    # 初始化 客户集 和 边缘服务器集
    clients = []
    for i in range(args.num_clients):
        # 初始化客户端
        clients.append(Client(id=i,
                              train_loader=train_loaders[i],
                              test_loader=test_loaders[i],
                              args=args,
                              device=device))

    initilize_parameters = list(clients[0].model.shared_layers.parameters())
    nc = len(initilize_parameters)
    for client in clients:
        user_parameters = list(client.model.shared_layers.parameters())
        for i in range(nc):
            user_parameters[i].data[:] = initilize_parameters[i].data[:]

    # Initialize edge server and assign clients to the edge server
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges

    if args.iid == -2:
        if args.edgeiid == 1:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_iid(num_edges=args.num_edges,
                                                    clients=clients,
                                                    client_class_dis=client_class_dis)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     client_class_dis=client_class_dis)
        else:
            # This is randomly assign the clients to edges
            for i in range(args.num_edges):
                if args.active_mapping == 1:
                    # 根据映射关系进行选择
                    selected_cids = mapping[str(i)]
                else:
                    # Randomly select clients and assign them
                    if i == args.num_edges - 1:  # 客户端数非边缘数的整数倍情况
                        selected_cids = cids
                    else:
                        selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
                print(f"Edge {i} has clients {selected_cids}")

                if args.attack_flag == 1: # 如果开启了模型攻击，就要构造每个edge的scids
                    selfish_cids = attack_mapping[str(i)]
                else:  selfish_cids = []
                print(f"Edge {i} has selfish clients {selfish_cids}")

                cids = list(set(cids) - set(selected_cids))
                edges.append(Edge(id=i, cids=selected_cids,
                                  shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                                  scids=selfish_cids, share_dataloader=share_loaders[i]))

                # 注册客户信息并按需把共享数据集给到客户端
                for cid in selected_cids:
                    edges[i].client_register(clients[cid])

                edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
                p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                                list(edges[i].sample_registration.values())]
    else:
        # This is randomly assign the clients to edges
        for i in range(args.num_edges):
            if args.active_mapping == 1:
                # 根据映射关系进行选择
                selected_cids = mapping[str(i)]
            else:
                # Randomly select clients and assign them
                if i == args.num_edges - 1:  # 客户端数非边缘数的整数倍情况
                    selected_cids = cids
                else:
                    selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            print(f"Edge {i} has clients {selected_cids}")
            cids = list(set(cids) - set(selected_cids))
            if args.attack_flag == 1:  # 如果开启了模型攻击，就要构造每个edge的scids
                selfish_cids = attack_mapping[str(i)]
            else:
                selfish_cids = []
            print(f"Edge {i} has selfish clients {selfish_cids}")
            edges.append(Edge(id=i, cids=selected_cids,
                              shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                              scids=selfish_cids, share_dataloader=share_loaders[i]))

            # 注册客户信息并按需把共享数据集给到客户端
            for cid in selected_cids:
                edges[i].client_register(clients[cid])

            edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
            p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                            list(edges[i].sample_registration.values())]

    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
              list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    # New an NN model for testing error
    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)

    # 开始训练
    # accs_edge_avg = []  # 记录云端的平均边缘测试精度
    # losses_edge_avg = []  # 记录云端的平均边缘损失
    accs_cloud = [0.0]  # 记录每轮云端聚合的精度
    times = [0]  # 记录每个云端轮结束的时间戳
    # 获取初始时间戳（训练开始时）
    start_time = time.time()
    for num_comm in tqdm(range(args.num_communication)):  # 云聚合
        cloud.refresh_cloudserver()
        [cloud.edge_register(edge=edge) for edge in edges]
        all_loss_sum = 0.0
        all_acc_sum = 0.0
        print(f"云端更新   第 {num_comm} 轮")
        for num_edgeagg in range(args.num_edge_aggregation):  # 边缘聚合
            print(f"边缘更新   第 {num_edgeagg} 轮")
            # 多线程的边缘迭代
            edge_threads = []
            edge_loss = [0.0] * len(edges)
            edge_sample = [0] * len(edges)
            for edge in edges:
                edge_thread = Thread(target=process_edge, args=(edge, clients, args, device, edge_loss, edge_sample))
                edge_threads.append(edge_thread)
                edge_thread.start()
            for edge_thread in edge_threads:
                edge_thread.join()
            # 统计边缘迭代的损失和样本
            total_samples = sum(edge_sample)
            if total_samples > 0:
                all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / total_samples
                all_loss_sum += all_loss
            else:
                print("Warning: Total number of samples is zero. Cannot compute all_loss.")
            print("train loss per edge on all samples: {}".format(edge_loss))

        print(models_are_equal(edges[0].shared_state_dict, edges[1].shared_state_dict))
        # 开始云端聚合
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        print(f"Cloud 聚合")
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)
        global_nn.load_state_dict(state_dict=copy.deepcopy(cloud.shared_state_dict))
        global_nn.train(False)

        # 云端测试
        print(f"Cloud 测试")
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        avg_acc_v = correct_all_v / total_all_v  # 测试精度
        print('Cloud Valid Accuracy {}'.format(avg_acc_v))
        # 在轮次结束时记录相对于开始时间的时间差, 记录云端轮的测试精度
        times.append(time.time() - start_time)
        accs_cloud.append(avg_acc_v)

    # 画出云端的精度-时间曲线图
    plt.plot(times, accs_cloud, marker='v', color='r', label="HierFL")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Test Model Accuracy')
    plt.title('Test Accuracy over Time')
    plt.show()


def train_client(client, edge, num_iter, device, return_dict, client_id):
    # print(f"Client {client.id} 本地迭代开始")
    # 如果设备是GPU，则设置相应的CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")  # 确保在每个线程中设置GPU
    # 客户端与边缘服务器同步
    edge.send_to_client(client)
    client.sync_with_edgeserver()
    # 执行本地迭代
    client_loss = client.local_update(num_iter=num_iter, device=device)
    # 将迭代后的模型发送回边缘服务器
    client.send_to_edgeserver(edge)
    # 存储结果
    return_dict[client_id] = client_loss
    # print(f"Client {client.id} 本地迭代结束")

def process_edge(edge, clients, args, device, edge_loss, edge_sample):
    # 一次边缘迭更新 = n个本地迭代+ 一次边缘聚合
    # print(f"Edge {edge.id} 边缘更新开始")
    # 使用多线程进行客户迭代
    threads = []
    return_dict = {}  # 在线程中，可以直接使用普通字典
    for selected_cid in edge.cids:
        client = clients[selected_cid]
        thread = Thread(target=train_client,
                        args=(client, edge, args.num_local_update, device, return_dict, selected_cid))
        threads.append(thread)
        thread.start()
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    # 边缘聚合
    # print(f"Edge {edge.id} 边缘聚合开始")
    edge.aggregate(args, device)
    # print(f"Edge {edge.id} 边缘聚合结束")
    # 更新边缘训练损失
    edge_loss[edge.id] = sum(return_dict.values())
    edge_sample[edge.id] = sum(edge.sample_registration.values())
    # print(f"Edge {edge.id} 边缘更新结束")


def main():
    args = args_parser()
    print(args.dataset_root)
    HierFAVG(args)


if __name__ == '__main__':
    main()
