# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
import random
from math import log

from torch.autograd import Variable
import torch
from torch.utils.data import ConcatDataset, DataLoader

from average import models_are_equal
from models.initialize_model import initialize_model
import copy


class Client():

    def __init__(self, id, train_loader, test_loader, args, device):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)  # 如果是gan类型
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        # self.batch_size = args.batch_size
        self.epoch = 0

        self.p = args.probability  # 给到客户的概率p
        self.mode = args.mode  # 决定客户如何上传模型
        self.train_loss = 0.0  # 本地训练损失

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                # print(inputs.shape)
                # print(labels.shape)
                # print("学习了样本label", labels)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels, device=device)
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch=self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        self.train_loss = loss  # 记录本轮训练的本地损失

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, matched_edge, all_edges):
        if self.mode == 0:  # 总是发送给匹配的 server
            matched_edge.receive_from_client(client_id=self.id,
                                             cshared_state_dict=copy.deepcopy(
                                                 self.model.shared_layers.state_dict()),
                                             train_loss=self.train_loss, train_sample_num=len(self.train_loader.dataset))
        elif self.mode == 1:  # 根据概率 p 发送给匹配的 server
            if random.random() <= self.p:
                matched_edge.receive_from_client(client_id=self.id,
                                                 cshared_state_dict=copy.deepcopy(
                                                     self.model.shared_layers.state_dict()),
                                                 train_loss=self.train_loss, train_sample_num=len(self.train_loader.dataset))
            else:
                print(f"客户{self.id} 不发送给边缘服务器{matched_edge.id}")
        else:  # 模式 2，如果不发送给匹配的 server，随机选择一个其他的 server 发送
            if random.random() <= self.p:
                matched_edge.receive_from_client(client_id=self.id,
                                                 cshared_state_dict=copy.deepcopy(
                                                     self.model.shared_layers.state_dict()),
                                                 train_loss=self.train_loss, train_sample_num=len(self.train_loader.dataset))
            else:
                # 从剩余的 edge servers 中随机选择一个
                remaining_servers = [es for es in all_edges if es.id != matched_edge.id]
                selected_server = random.choice(remaining_servers)
                selected_server.receive_from_client(client_id=self.id,
                                                    cshared_state_dict=copy.deepcopy(
                                                        self.model.shared_layers.state_dict()),
                                                    train_loss=self.train_loss, train_sample_num=len(self.train_loader.dataset))
                print(f"客户{self.id} 发送给另选的边缘服务器{selected_server.id}")

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # print(models_are_equal(self.receiver_buffer, self.model.shared_layers.state_dict()))
        self.model.shared_layers.load_state_dict(self.receiver_buffer)
        # print(set(self.model.shared_layers.state_dict().keys()) == set(self.receiver_buffer.keys()))
        self.model.update_model(self.receiver_buffer)
