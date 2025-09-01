'''
@contact:xind2023@mail.ustc.edu.cn
@time:2025/9/1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CBCLLoss(nn.Module):
    def __init__(self,cls_num_list=None, proxy_num_list = None, class_difficulty =  None, alpha = 1, p = 2, beta = (1 - 1/(1+math.exp(-1)))/30, temperature=0.1,):
        super(CBCLLoss, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list
        self.proxy_num_list = proxy_num_list

        m_list = 1.0 / np.sqrt(cls_num_list)
        m_list = m_list * (6 / max(m_list))
        self.m_list = m_list
        self.class_difficulty = class_difficulty
        self.alpha = alpha
        self.p = float(p)
        self.beta = beta
        p = float(p)
        self.weights = alpha * self.class_difficulty ** p + beta

    def forward(self, proxy, features, targets, class_difficulty):

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)

        # get proxy labels
        targets_proxy = torch.empty((0, 1), dtype=torch.int64)
        for i, num in enumerate(self.proxy_num_list):
            tmp_targets = torch.full([int(num), 1], i)
            targets_proxy = torch.cat((targets_proxy, tmp_targets), dim=0)

        targets_proxy = targets_proxy.view(-1, 1).to(device)

        # get labels of features and proxies
        targets = torch.cat([targets.repeat(2, 1), targets_proxy], dim=0).to('cpu')
        batch_cls_count = torch.eye(len(self.cls_num_list))[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2 + int(np.array(self.proxy_num_list).sum())).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # get similarity matrix
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, proxy], dim=0)
        logits = features.mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class averaging
        exp_logits = torch.exp(logits) * logits_mask

        self.weights = self.alpha * class_difficulty ** self.p + self.beta
        self.m_list = torch.tensor(self.m_list, dtype=torch.float32, device='cuda')
        self.weights = torch.tensor(self.weights, dtype=torch.float32, device='cuda')
        batch_cls_count = batch_cls_count.to(device)

        max_m = (max(self.cls_num_list) / min(self.cls_num_list)) // 10 + 2
        self.weights = (self.weights) * ((max_m) / max(self.weights))

        self.proxy_num_list = torch.tensor(self.proxy_num_list, dtype=torch.float32, device='cuda')
        self.proxy_num_list = self.proxy_num_list.cpu().numpy()


        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size + int(np.array(self.proxy_num_list).sum()), 2 * batch_size + int(np.array(self.proxy_num_list).sum())) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        # get loss
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss

