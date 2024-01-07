# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
from average import average_weights, models_are_equal


class Cloud():

    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)  # 记录edge的总样本量
        self.sample_registration[edge.id] = sum(edge.sample_registration.values())
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [(self.sample_registration[eid], dict) for eid, dict in self.receiver_buffer.items()
                         if self.sample_registration[eid] > 0]
        if len(received_dict) != 0: # 只有当边缘模型个数>=1时，才进行聚合
            self.shared_state_dict = average_weights(received_dict)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None

