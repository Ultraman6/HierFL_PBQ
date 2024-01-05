import copy
import torch
from torch import nn
# 权重平均聚合
def average_weights(w_locals):
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

# def average_weights(w, s_num):
#     total_sample_num = sum(s_num)
#     scale_factors = [num / total_sample_num for num in s_num]
#
#     # 初始化平均权重
#     w_avg = {k: torch.zeros_like(w[0][k]) for k in w[0].keys()}
#
#     # 累加每个客户端的权重
#     for i, w_client in enumerate(w):
#         for k in w_client.keys():
#             w_avg[k] += w_client[k] * scale_factors[i]
#
#     return w_avg

def models_are_equal(model_a_state_dict, model_b_state_dict):
    for key in model_a_state_dict:
        if key not in model_b_state_dict:
            return False
        if not torch.equal(model_a_state_dict[key], model_b_state_dict[key]):
            return False
    return True
