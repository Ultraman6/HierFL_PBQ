# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
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
        self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        # self.batch_size = args.batch_size
        self.epoch = 0
        # self.share_dataloader = None  # 添加共享数据加载器属性

    def combine_share_data(self, share_dataloader):
            # 合并数据集
            combined_dataset = ConcatDataset([self.train_loader.dataset, share_dataloader.dataset])
            # 创建新的数据加载器
            self.train_loader = DataLoader(combined_dataset, batch_size=self.train_loader.batch_size,
                                           shuffle=True, num_workers=self.train_loader.num_workers, pin_memory=True)
            # print(len(self.train_loader.dataset))

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print("学习了样本label", labels)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
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
            self.model.exp_lr_sheduler(epoch = self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

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

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id=self.id,
                                       cshared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict()))

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # print(models_are_equal(self.receiver_buffer, self.model.shared_layers.state_dict()))
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)