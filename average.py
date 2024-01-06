import copy
from collections import OrderedDict

import torch
from torch import nn
# 权重平均聚合

def average_weights(w_locals):
    if is_gan_model(w_locals):
        return average_weights_gan(w_locals)
    elif is_rnn_parameters(w_locals):
        return average_weights_rnn(w_locals)
    else:
        return average_weights_std(w_locals)


def average_weights_std(w_locals):
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num
    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params

def is_gan_model(w_locals):
    _, sample_weights = w_locals[0]
    return "generator" in sample_weights and "discriminator" in sample_weights

def is_rnn_parameters(w_locals):
    # 假设 w_locals 中的每个元素都是 (sample_num, model_parameters) 的元组
    _, model_parameters = w_locals[0]
    # 检查参数名称中是否包含 'lstm'
    for param_name in model_parameters.keys():
        if 'lstm' in param_name:
            return True
    return False

def average_weights_gan(w_locals):
    # 计算总的训练样本数量
    training_num = sum(sample_num for sample_num, _ in w_locals)

    # 初始化加权平均后的生成器和判别器参数
    _, first_weights = w_locals[0]
    averaged_params_gen = {k: torch.zeros_like(v, dtype=torch.float) for k, v in first_weights['generator'].items()}
    averaged_params_disc = {k: torch.zeros_like(v, dtype=torch.float) for k, v in first_weights['discriminator'].items()}

    # 对所有客户端的参数进行加权平均
    for sample_num, model_weights in w_locals:
        weight = float(sample_num) / training_num
        for k in averaged_params_gen.keys():
            averaged_params_gen[k] += model_weights['generator'][k].float() * weight
        for k in averaged_params_disc.keys():
            averaged_params_disc[k] += model_weights['discriminator'][k].float() * weight

    return {'generator': averaged_params_gen, 'discriminator': averaged_params_disc}




def average_weights_rnn(updates):
    avg_param = OrderedDict()
    total_weight = 0.
    for (client_samples, client_model) in updates:
        total_weight += client_samples
        for name, param in client_model.items():
            if name not in avg_param:
                avg_param[name] = client_samples * param
            else:
                avg_param[name] += client_samples * param

    for name in avg_param:
        avg_param[name] = avg_param[name] / total_weight
    return copy.deepcopy(avg_param)

# def average_weights(w_locals):
#     """
#     基于样本量对 RNN 模型的权重进行加权平均。
#     w_locals - 包含 (sample_num, model) 对的列表。
#     """
#     # 计算总样本量
#     total_samples = sum(sample_num for sample_num, _ in w_locals)
#
#     # 初始化加权平均后的参数字典
#     averaged_params = OrderedDict()
#
#     for name, param in w_locals[0][1].named_parameters():
#         averaged_param = torch.zeros_like(param.data)
#
#         for sample_num, model in w_locals:
#             local_params = dict(model.named_parameters())
#
#             # 将 w 转换为张量
#             w_tensor = torch.tensor(sample_num / total_samples).to(param.data.device)
#
#             if 'lstm.weight_hh' in name or 'lstm.weight_ih' in name:
#                 # 使用张量 w_tensor
#                 averaged_param += local_params[name].data * w_tensor.unsqueeze(-1).unsqueeze(-1)
#             else:
#                 averaged_param += local_params[name].data * w_tensor
#
#         averaged_params[name] = averaged_param
#
#     return averaged_params

def models_are_equal(model_a_state_dict, model_b_state_dict):
    for key in model_a_state_dict:
        if key not in model_b_state_dict:
            return False
        if not torch.equal(model_a_state_dict[key], model_b_state_dict[key]):
            return False
    return True

def gan_state_dicts_are_equal(state_dict_a, state_dict_b):
    # 检查是否包含 "generator" 和 "discriminator" 键
    if "generator" not in state_dict_a or "discriminator" not in state_dict_a:
        return False
    if "generator" not in state_dict_b or "discriminator" not in state_dict_b:
        return False

    # 比较生成器的状态字典
    if not models_parameters_are_equal(state_dict_a["generator"], state_dict_b["generator"]):
        return False

    # 比较鉴别器的状态字典
    if not models_parameters_are_equal(state_dict_a["discriminator"], state_dict_b["discriminator"]):
        return False

    return True

def models_parameters_are_equal(model_a_state_dict, model_b_state_dict):
    # 检查两个状态字典是否具有相同的键集合
    if set(model_a_state_dict.keys()) != set(model_b_state_dict.keys()):
        return False

    # 检查每个键对应的张量是否相等
    for key in model_a_state_dict:
        if not torch.equal(model_a_state_dict[key], model_b_state_dict[key]):
            return False

    return True
