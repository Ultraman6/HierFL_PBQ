import argparse
import json
import os

import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'cinic10',
        help = 'name of the dataset: mnist, cifar10, femnist, synthetic, cinic10, shakespare, CelebA'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'resnet18_YWX',
        help='name of model. mnist: logistic, lenet, cnn; '
             'cifar10、cinic10: resnet18, resnet18_YWX, cnn_complex; '
             'femnist: logistic, lenet, cnn; synthetic: lr; '
             'shakespare: RNN; CelebA: GAN'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 3,
        help = 'input channels. femnist:1, mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 62,
        help = 'output channels. femnist:62'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--train_batch_size',
        type = int,
        default = 8,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--test_batch_size',
        type = int,
        default = 16,
        help = 'batch size when tested on client'
    )
    parser.add_argument(
        '--num_workers',
        type = int,
        default = 4,
        help = 'numworks for dataloader'
    )
    parser.add_argument(
        '--test_ratio',
        type = int,
        default = 0.1,
        help = 'ratio of test dataset'
    )
    # -------------云聚合轮次、边缘聚合轮次、本地更新轮次
    parser.add_argument(
        '--num_communication',
        type = int,
        default=100,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=2,
        help = 'number of edge aggregation (K_2)'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='number of local update (K_1)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '0',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0.9,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 5e-3,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = -3,
        help = 'distribution of the data, 1,0,-1,-2 分别表示iid同大小、niid同大小、iid不同大小、niid同大小且仅一类(one-class)'
    )
    # 缓解niid策略——每个edge共享数据给足下客户
    parser.add_argument(
        '--niid_share',
        type = int,
        default = 0,
        help = '1 表示开启共享niid缓解， 0表示关闭. 该参数将直接影响初始数据划分和训练前给客户并入数据'
    )
    parser.add_argument(
        '--edgeiid',   # 只有在客户数 = 10倍的边缘数才生效
        type=int,
        default = -1,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) -1 (none) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )
    # -------------客户端数、边缘服务器数、客户端训练样本量
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 30,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 3,
        help= 'number of edges'
    )

    # editer: Extrodinary 20231215
    # 定义clients及其分配样本量的关系
    parser.add_argument(
        '--self_sample',
        default= -1,
        type=int,
        help='>=0: set sample of each client， -1: auto samples'
    )

    # 将映射关系转换为JSON格式，主键个数必须等于num_edges，value为-1表示all samples
    sample_mapping_json = json.dumps({
        "0": 3000,
        "1": 2000,
        "2": 5000,
        "3": 10000,
        "4": -1,
    })
    parser.add_argument(
        '--sample_mapping',
        type = str,
        default = sample_mapping_json,
        help = 'mapping of clients and their samples'
    )

    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )

    # 设置数据集的根目录为家目录下的 train_data 文件夹
    dataset_root = os.path.join(os.path.expanduser('~'), 'train_data')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    parser.add_argument(
        '--dataset_root',
        type = str,
        default = dataset_root,
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 1,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default = 1,
        help='under artificial non-iid distribution, the classes per client'
    )

    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )

    parser.add_argument(
        '--test_on_all_samples',
        type = int,
        default = 1,
        help = '1 means test on all samples, 0 means test samples will be split averagely to each client'
    )
    # 定义edges及其下属clients的映射关系
    parser.add_argument(
        '--active_mapping',
        type = int,
        default = 1,
        help = '1 means mapping is active, 0 means mapping is inactive'
    )
    mapping = {
        "0": [0, 1, 2, 13, 14, 15, 18],
        "1": [3, 4, 5, 10, 11, 16],
        "2": [6, 7, 8, 9, 12, 17, 19],
    }
    # 将映射关系转换为JSON格式
    mapping_json = json.dumps(mapping)
    parser.add_argument(
        '--mapping',
        type = str,
        default = mapping_json,
        help = 'mapping of edges and their clients'
    )

    # synthetic数据集用参数
    parser.add_argument(
        '--alpha',
        type = int,
        default = 1,
        help = 'means the mean of distributions among clients'
    )
    parser.add_argument(
        '--beta',
        type = float,
        default = 1,
        help = 'means the variance  of distributions among clients'
    )
    parser.add_argument(
        '--dimension',
        type = int,
        default = 60,
        help = '1 means mapping is active, 0 means mapping is inactive'
    )
    parser.add_argument(
        '--num_class',
        type = int,
        default = 10,
        help = '1 means mapping is active, 0 means mapping is inactive'
    )


    # 模型攻击参数
    parser.add_argument(
        '--attack_flag',
        type = int,
        default = 1,
        help = 'trigger of attack,1 means on 0 means off'
    )
    attack_mapping = {
        "0": [0, 2],
        "1": [3],
        "2": [8],
    }
    # 将映射关系转换为JSON格式
    attack_mapping_json = json.dumps(attack_mapping)
    parser.add_argument(
        '--attack_mapping',
        type = str,
        default = attack_mapping_json,
        help = 'mapping of edges and their selfish clients'
    )
    parser.add_argument(
        '--attack_mode',
        type = str,
        default = "flip",
        help = 'mode of attack, such as: zero、random、flip'
    )

    # 新~数据划分方法
    parser.add_argument(
        '--partition',
        type = str,
        default = 'noniid-labeldir',
        help = '划分类型，homo、 noniid-labeldir、 noniid-#label1、 iid-diff-quantity, '
               '其中label后的数字表示每个客户的类别数'
    )
    parser.add_argument(
        '--beta_new',
        type = float,
        default = 10,
        help = 'dir分布的超参数'
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
