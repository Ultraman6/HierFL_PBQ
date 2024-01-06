# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)
import json
import time
from threading import Thread

from matplotlib import pyplot as plt

from models.celeba_gan import GAN, Generator, Discriminator
from models.shakespare_rnn import RNNModel
from models.synthetic_logistic import LogisticRegression_SYNTHETIC
from options import args_parser
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
            edge_class[i].append(list(label))
    print(f'class distribution among edge {edge_class}')


def initialize_edges_iid(num_edges, clients, client_class_dis, mode):
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
        edges.append(Edge(id=eid, cids=assigned_clients_idxes, shared_layers=copy.deepcopy(clients[0].model.shared_layers),mode=mode))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        p_clients[eid] = [sample / float(sum(edges[eid].sample_registration.values()))
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
    edges.append(Edge(id=eid, cids=assigned_clients_idxes, shared_layers=copy.deepcopy(clients[0].model.shared_layers),mode=mode))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    p_clients[eid] = [sample / float(sum(edges[eid].sample_registration.values()))
                      for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients


def initialize_edges_niid(num_edges, clients, client_class_dis, mode):
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
        edges.append(Edge(id=eid, cids=assigned_clients_idxes,shared_layers=copy.deepcopy(clients[0].model.shared_layers),mode=mode))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        p_clients[eid] = [sample / float(sum(edges[eid].sample_registration.values()))
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
    edges.append(Edge(id=eid,cids=assigned_clients_idxes,shared_layers=copy.deepcopy(clients[0].model.shared_layers),mode=mode))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    p_clients[eid] = [sample / float(sum(edges[eid].sample_registration.values()))
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


def fast_all_clients_test_gan(v_test_loader, global_gan, device):
    global_gan.generator.eval()
    global_gan.discriminator.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data in v_test_loader:
            real_images, _ = data
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            total_predictions += 2 * batch_size  # Counting both real and fake images

            # Generate fake images
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            fake_images = global_gan.generator(noise)

            # Compute loss for real and fake images
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_output = global_gan.discriminator(real_images)
            fake_output = global_gan.discriminator(fake_images)

            # Calculate 'accuracy'
            correct_predictions += ((real_output > 0.5) == real_labels).sum().item()
            correct_predictions += ((fake_output < 0.5) == fake_labels).sum().item()

    return correct_predictions, total_predictions


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
    elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
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
    elif args.dataset == 'shakespare':
        if args.model == 'rnn':
            global_nn = RNNModel()
    elif args.dataset == 'celeba':
        if args.model == 'gan':
            global_nn = GAN(Generator(), Discriminator())
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
    train_loaders, test_loaders, v_test_loader = get_dataloaders(args)

    if args.show_dis and args.dataset != 'shakespare':
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

        print("Cloud valid data size: {}".format(len(v_test_loader.dataset)))
        valid_data_distribution = show_distribution(v_test_loader, args)
        print("Cloud valid data distribution: {}".format(valid_data_distribution))

    # 读取mapping配置信息
    mapping = None
    if args.active_mapping == 1:
        # 提取映射关系参数并将其解析为JSON对象
        mapping = json.loads(args.mapping)

    # 初始化 客户集 和 边缘服务器集
    clients = []
    for i in range(args.num_clients):
        # 初始化客户端
        clients.append(Client(id=i,
                              train_loader=train_loaders[i],
                              test_loader=test_loaders[i],
                              args=args,
                              device=device))


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
                                                    client_class_dis=client_class_dis, mode=args.mode)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     client_class_dis=client_class_dis, mode=args.mode)
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

                cids = list(set(cids) - set(selected_cids))    # 这里是GAN也没关系，反正都是模型类
                edges.append(Edge(id=i,cids=selected_cids, shared_layers=copy.deepcopy(clients[0].model.shared_layers),mode=args.mode))

                # 注册客户信息并按需把共享数据集给到客户端
                for cid in selected_cids:
                    edges[i].client_register(clients[cid])

                all_trainsample_num = sum(edges[i].sample_registration.values())
                p_clients[i] = [sample / float(all_trainsample_num) for sample in
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
            edges.append(Edge(id=i,cids=selected_cids,shared_layers=copy.deepcopy(clients[0].model.shared_layers),mode=args.mode))

            # 注册客户信息并按需把共享数据集给到客户端
            for cid in selected_cids:
                edges[i].client_register(clients[cid])

            all_trainsample_num = sum(edges[i].sample_registration.values())
            p_clients[i] = [sample / float(all_trainsample_num) for sample in
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
        print(f"云端更新   第 {num_comm} 轮")
        for num_edgeagg in range(args.num_edge_aggregation):  # 边缘聚合
            print(f"边缘更新   第 {num_edgeagg} 轮")
            # 多线程的边缘迭代
            edge_threads = []
            edge_loss = [0.0] * len(edges)
            edge_sample = [0] * len(edges)
            for edge in edges: # 多线程第一阶段：edge并行训练
                edge_thread = Thread(target=process_edge_train, args=(edge, clients, args.num_local_update, device, edges))
                edge_threads.append(edge_thread)
                edge_thread.start()
            for edge_thread in edge_threads:
                edge_thread.join()
            edge_threads.clear()   # 清空edge训练的线程，准备装入edge聚合的线程
            for edge in edges:  # 多线程第二阶段：edge并行聚合
                edge_thread = Thread(target=process_edge_agg, args=(edge, edge_loss, edge_sample))
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

        # print(models_are_equal(edges[0].shared_state_dict, edges[1].shared_state_dict))
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
        if args.model == 'gan':
            correct_all_v, total_all_v = fast_all_clients_test_gan(v_test_loader, global_nn, device)
        else:
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


def train_client(client, edge, num_iter, device, all_edges):
    # print(f"Client {client.id} 本地迭代开始")
    # 如果设备是GPU，则设置相应的CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")  # 确保在每个线程中设置GPU
    # 客户端与边缘服务器同步
    edge.send_to_client(client)
    client.sync_with_edgeserver()

    # 执行本地迭代
    client.local_update(num_iter=num_iter, device=device)
    # 将迭代后的模型发送回边缘服务器
    client.send_to_edgeserver(edge, all_edges)
    # 存储结果
    # print(f"Client {client.id} 本地迭代结束")


def process_edge_train(edge, clients, num_local_update, device, all_edges):
    edge.refresh_edgeserver()  # 清空上一轮的客户本地模型参数、本地样本量、本地训练损失
    # 一次边缘迭更新 = n个本地迭代 + 一次边缘聚合
    print(f"Edge {edge.id} 的配对客户 {edge.id_registration}")
    # 使用多线程进行客户迭代
    threads = []
    for selected_cid in edge.id_registration:
        client = clients[selected_cid]
        thread = Thread(target=train_client,
                        args=(client, edge, num_local_update, device, all_edges))
        threads.append(thread)
        thread.start()
    # 等待所有线程完成
    for thread in threads:
        thread.join()

def process_edge_agg(edge, edge_loss, edge_sample):
    # 边缘聚合
    edge.aggregate()
    # 更新边缘训练损失
    if len(edge.train_losses) != 0:
        edge_loss[edge.id] = sum(edge.train_losses) / len(edge.train_losses)
    else: # 如果没有客户上传模型，损失为0
        edge_loss[edge.id] = 0.0
    edge_sample[edge.id] = sum(edge.sample_registration)

def main():
    args = args_parser()
    print(args.dataset_root)
    HierFAVG(args)


if __name__ == '__main__':
    main()
