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

    def __init__(self, id, cids, shared_layers, mode):
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
        self.cids = cids  # 保存初始的客户id集
        self.receiver_buffer = {}  # 暂存客户端上传的模型参数
        self.shared_state_dict = {}  # 全局模型参数
        self.id_registration = []  # 当前轮次的客户
        self.sample_registration = {}
        self.shared_state_dict = shared_layers.state_dict()
        self.train_losses = []
        self.mode = mode  # edge需要mode判断对不上传的客户是否保留注册信息

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.sample_registration.clear()
        self.train_losses.clear()
        if self.mode != 2:
            self.id_registration = self.cids
        return None

    def client_register(self, client):
        if client.id not in self.id_registration:
            self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)

    def receive_from_client(self, client_id, cshared_state_dict, train_loss, train_sample_num):
        self.receiver_buffer[client_id] = cshared_state_dict  # 客户上传模型同时上传，本地损失，本地样本数
        self.train_losses.append(train_loss)
        self.sample_registration[client_id] = train_sample_num
        return None

    def aggregate(self):
        # Check if the attack is enabled and perform the attack if necessary
        self.id_registration = list(self.receiver_buffer.keys())  # 聚合时复盘哪些客户给我传了模型
        print('edge{} 接收的客户：{}'.format(self.id, str(self.id_registration)))
        # print('edge{} 接收的客户的损失：{}'.format(self.id, str(self.train_losses)))
        if len(self.receiver_buffer) != 0:
            received_dict = [(self.sample_registration[cid], dict) for cid, dict in self.receiver_buffer.items()]
            self.shared_state_dict = average_weights(received_dict)
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



