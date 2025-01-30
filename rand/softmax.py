import torch

def softmax(x):
    x_max = torch.max(x)  # 1st pass
    exp_x = torch.exp(x - x_max)  # 2nd pass
    denominator = exp_x.sum(exp_x)  # 3rd pass
    return exp_x / denominator



