'''
@contact:xind2023@mail.ustc.edu.cn
@time:2025/9/1
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def safe_cross_entropy(output, target, weight=None, s=1.0):
    scaled_output = output * s
    log_probabilities = F.log_softmax(scaled_output, dim=1)
    new_log_probabilities = log_probabilities.clone()
    for col in range(len(log_probabilities[0])):
        for row in range(len(log_probabilities)):
            new_log_probabilities[row, col] = 1 * log_probabilities[row, col]

    log_probabilities = new_log_probabilities

    target_one_hot = torch.zeros_like(log_probabilities).scatter_(1, target.view(-1, 1), 1)
    log_prob_for_target = (target_one_hot * log_probabilities).sum(dim=1)

    ce_loss = -log_prob_for_target.mean()

    if weight is not None:
        ce_loss = ce_loss * weight.mean()

    return ce_loss



class CBAMLoss(nn.Module):

    def __init__(self, cls_num_list, class_difficulty, alpha, p, beta, E1, E2, max_m=0.5, weight=None, s=30):
        super(CBAMLoss, self).__init__()

        max_m = -math.log(min(cls_num_list)/sum(cls_num_list))-0.165745444183859

        cls_p_list = 1.0 / np.sqrt(cls_num_list)
        cls_p_list = torch.cuda.FloatTensor(cls_p_list)

        m_list = max_m * cls_p_list / max(cls_p_list)

        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.class_difficulty = class_difficulty
        self.alpha = alpha
        self.p = float(p)
        self.beta = beta
        self.max_m = max_m
        assert s > 0
        self.s = s
        self.E1 = E1
        self.E2 = E2

    def forward(self, x, targets, epoch, class_difficulty):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        target = targets

        self.weights = self.alpha * class_difficulty ** self.p + self.beta

        self.m_list = torch.tensor(self.m_list, dtype=torch.float32, device='cuda')

        self.weights = torch.tensor(self.weights, dtype=torch.float32, device='cuda')

        self.weights = self.weights * ((self.max_m/1) / max(self.weights))

        if (epoch >= self.E1 and epoch < self.E2):
            ee = (epoch-self.E1)/(self.E2-self.E1)
        if epoch >= self.E2:
            ee = 1
        
        # lin
        # if (epoch >= self.E1 and epoch < self.E2):
        #     ee = (epoch-self.E1)/(self.E2-self.E1)
        #     ee = (3*(ee**2)-2*(ee**3))
        # if epoch >= self.E2:
        #     ee = 1

        # exp
        if epoch < self.E1:
            self.m_list1 = self.m_list
        if epoch >= self.E1:
            self.m_list1 = self.m_list + ((self.weights)*ee)/2

        self.m_list1 = torch.tensor(self.m_list1, dtype=torch.float32, device='cuda')
        index_float = torch.tensor(index, dtype=torch.float32, device='cuda')
        batch_m = torch.matmul(self.m_list1[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)

        return safe_cross_entropy(output, target, weight=self.weight)