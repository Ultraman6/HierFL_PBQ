import random

import numpy as np
import torch

from average import average_weights


def sample_some_clients(client_num, sampled_client_num):
    return random.sample(range(client_num), sampled_client_num)

def is_weight_param(k):
    return (
            "running_mean" not in k
            and "running_var" not in k
            and "num_batches_tracked" not in k
    )

def perform_byzantine_attack(raw_client_grad_dict, byzantine_client_ids, attack_mode, device):

    print(f"byzantine_idxs={byzantine_client_ids}")
    if attack_mode == "zero":
        return attack_zero_mode(raw_client_grad_dict, byzantine_client_ids, device)
    elif attack_mode == "random":
        return attack_random_mode(raw_client_grad_dict, byzantine_client_ids, device)
    elif attack_mode == "flip":
        return attack_flip_mode(raw_client_grad_dict, byzantine_client_ids, device)
    else:
        raise NotImplementedError("Attack mode not implemented!")

def attack_zero_mode(model_list, byzantine_idxs, device):
    new_model_list = []
    for cid in model_list.keys():
        if cid in byzantine_idxs:
            num, local_model_params = model_list[cid]
            for k in local_model_params.keys():
                if is_weight_param(k):
                    local_model_params[k] = torch.zeros(local_model_params[k].size(), device=device)
            new_model_list.append((num, local_model_params))
            print("client {} 发动zero攻击".format(cid))
        else:
            new_model_list.append(model_list[cid])
    return new_model_list

def attack_random_mode(model_list, byzantine_idxs, device):
    new_model_list = []
    for cid in model_list.keys():
        if cid in byzantine_idxs:
            num, local_model_params = model_list[cid]
            for k in local_model_params.keys():
                if is_weight_param(k):
                    local_model_params[k] = torch.from_numpy(2 * np.random.random_sample(local_model_params[k].size()) - 1).float().to(device)
            new_model_list.append((num, local_model_params))
            print("client {} 发动random攻击".format(cid))
        else:
            new_model_list.append(model_list[cid])
    return new_model_list

def attack_flip_mode(model_list, byzantine_idxs, device):
    # 先聚合得到初始全局模型
    global_model = average_weights(list(model_list.values()))
    new_model_list = []
    for cid in model_list.keys():
        if cid in byzantine_idxs:
            num, local_model_params = model_list[cid]
            for k in local_model_params.keys():
                if is_weight_param(k):
                    local_model_params[k] = 2 * global_model[k].to(device) - local_model_params[k].to(device)
            new_model_list.append((num, local_model_params))
            print("client {} 发动flip攻击".format(cid))
        else:
            new_model_list.append(model_list[cid])
    return new_model_list
