# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights, models_are_equal


class Edge():

    def __init__(self, id, cids, shared_layers, mode, tao):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids  # 保存初始和上一轮绑定的客户id集; 当选项3情况下，会由于间谍客户而改变
        self.receiver_buffer = {}  # 暂存客户端上传的模型参数
        self.shared_state_dict = {}  # 全局模型参数
        self.id_registration = []  # 记录当前轮次的客户id，在非选项3情况下，充当cids使用

        self.sample_registration = {}
        self.all_sample_num = 0  # 保存上一轮次的总样本量,只在注册和复盘时更新
        self.shared_state_dict = shared_layers.state_dict()
        self.train_losses = []
        self.mode = mode  # edge需要mode判断对不上传的客户是否保留注册信息
        self.tao = tao # edge用聚合超参数
        # self.now_all_sample_num = 0   # 目前的总样本数

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.sample_registration.clear()
        self.train_losses.clear()
        # self.now_all_sample_num = 0  # 新的边缘轮开始需要清0
        if self.mode < 2:
            self.id_registration = self.cids
        return None

    def client_register(self, client):
        if client.id not in self.id_registration:
            self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        self.all_sample_num += self.sample_registration[client.id]

    # 新增逻辑，若为选项3，则此时客户可能上传间谍样本量(tao/s1),这时还不能更新总样本量，因为可能有间谍需要上一轮的总样本量
    def receive_from_client(self, client_id, cshared_state_dict, train_loss, train_sample_num): # 非绑定客户上传两个样本量一个参数：本地原始、原绑定e的总样本量， 原edge的id
        self.receiver_buffer[client_id] = cshared_state_dict  # 客户上传模型同时上传，本地损失，本地样本数
        self.train_losses.append(train_loss)
        self.sample_registration[client_id] = train_sample_num # 此时不关注train_sample_num的信息
        return None

    def aggregate(self, flag):
        self.id_registration = list(self.receiver_buffer.keys())  # 聚合时复盘哪些客户给我传了模型
        self.all_sample_num = 0  # 准备接受新的客户上传的样本量
        # 梯度非绑定用户上传的信息
        print('edge{} 接收的客户：{}'.format(self.id, str(self.id_registration)))
        if len(self.receiver_buffer) != 0:  # 当有客户上传模型时
            received_dict = []
            if self.mode == 3 and flag:  # 模式3，且非首边缘轮
                # 找出非绑定客户，并计算每个e下的间谍数量
                no_bind_cids = [cid for cid in self.id_registration if cid not in self.cids]
                eid_lengths = {}
                for ncid in no_bind_cids: # 遍历所有未绑定客户id
                    eid_lengths[self.sample_registration[ncid][1][0]] = 0
                for ncid in no_bind_cids: # 遍历所有未绑定客户id
                    eid_lengths[self.sample_registration[ncid][1][0]] += 1
                for cid, dict in self.receiver_buffer.items():
                    if cid in no_bind_cids:
                        received_dict.append((self.sample_registration[cid][1][1] * self.tao /
                                              eid_lengths[self.sample_registration[cid][1][0]], dict))
                        self.all_sample_num += self.sample_registration[cid][0]
                    else:
                        received_dict.append((self.sample_registration[cid], dict))
                        self.all_sample_num += self.sample_registration[cid]
            else:
                received_dict = [(self.sample_registration[cid], dict) for cid, dict in self.receiver_buffer.items()]
                self.all_sample_num = sum(self.sample_registration.values())  # 直接求和就是这一轮edge的总样本量(这地方没加value就是对键值求和，如果只有一个id=0的客户下面的聚合直接g)
            self.shared_state_dict = average_weights(received_dict)
        self.cids = self.id_registration

        return None

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        # print(models_are_equal(shared_state_dict, self.shared_state_dict))
        self.shared_state_dict = shared_state_dict
        return None



