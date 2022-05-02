import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_score(pos, neg):
    score_neg_mean = torch.mean(neg, dim=1, keepdim=True).half().cpu().detach().numpy()
    score_neg_var = torch.var(neg, dim=1, keepdim=True).half().cpu().detach().numpy()
    score_pos = pos.half().cpu().detach().numpy()
    # exp_neg_mean = torch.mean(torch.exp(neg / t), dim=1, keepdim=True).cpu().detach().numpy()
    # exp_pos_mean = torch.exp(pos / t).cpu().detach().numpy()

    if min(neg.shape) == 0:
        neg = torch.zeros_like(pos)
    I_neg = list(map(lambda x: x[0] > min(x[1]), zip(neg, pos)))
    I_pos = list(map(lambda x: max(x[0]) > x[1], zip(neg, pos)))
    count_neg = [i.tolist().count(True) for i in I_neg]
    count_pos = [i.tolist().count(True) for i in I_pos]
    count_neg = np.array(count_neg).reshape(len(count_neg), 1)
    count_pos = np.array(count_pos).reshape(len(count_pos), 1)

    score = np.concatenate((score_neg_mean, score_neg_var, count_neg, count_pos, score_pos), axis=1)
    return score, neg


class sMoCo(nn.Module):

    def __init__(self, encoder_q, encoder_k, dim_in, K=2000, t=0.07, momentum=0.99):
        super(sMoCo, self).__init__()
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dim_in = dim_in
        self.K = K 
        self.t = t 
        self.momentum = momentum
        self.weight = 0
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.memory_queue = F.normalize(torch.randn(self.K, self.dim_in).cuda(), dim=-1)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def get_sim(self, anchor, pos, flag=1):
        sims = []
        idx = anchor.size(0)
        for i in range(idx):
            if flag:
                sim = [torch.cosine_similarity(anchor[i], pos[i, j], dim=0) for j in range(pos.size(1))]
            else:
                sim = [torch.cosine_similarity(anchor[i], pos[j], dim=0) for j in range(pos.size(0))]
            sim = torch.stack(sim, 0)
            sims.append(sim)
        sims = torch.stack(sims, 0)
        return sims

    def anchor_mix_pos(self, anchor, pos, weight=0.2):
        pos_mix = []
        for i in range(anchor.size(0)):
            if len(pos.shape) == 3:
                new_pos = torch.stack([weight * anchor[i] + (1-weight) * pos[i, j, :] for j in range(pos.size(1))], dim=0)
            else:
                new_pos = torch.stack([weight * anchor[i] + (1-weight) * pos[i]], dim=0)
            pos_mix.append(new_pos)
        pos_mix = torch.stack(pos_mix, dim=0)
        return pos_mix

    def forward(self, x_q, x_k, batch_size=16, normalize=True, mode='all'):

        loss, _, _, query = self.encoder_q(x_q[0], x_q[1])
        if normalize:
            query = F.normalize(query, dim=1)

        # momentum update
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            _, _, _, key = self.encoder_k(x_k[0], x_k[1])
            idx = torch.randperm(key.size(0))
            key = key[torch.argsort(idx)]
            if normalize:
                key = F.normalize(key, dim=1)

        pos = key if batch_size == 8 else torch.stack(torch.split(key, 3, dim=0), dim=0)

        # get all pos similarity
        if batch_size == 16:
            sim_pos = self.get_sim(query, pos, flag=1)
            if mode == 'hard':                
                sim_pos_min = torch.min(sim_pos, dim=1)
                hard_pos = [torch.index_select(pos[i], 0, sim_pos_min[1][i]) for i in range(len(sim_pos_min[1]))]
                hard_pos = torch.stack(hard_pos, dim=0).squeeze(1)
                score_pos = torch.bmm(query.unsqueeze(dim=1), hard_pos.unsqueeze(dim=-1)).squeeze(dim=-1)  # N*1
            elif mode == 'all':
                score_pos = sim_pos
            elif mode == 'two':
                score_pos = torch.topk(sim_pos, 2, dim=1, largest=False)[0]
            elif mode == 'easy':
                sim_pos_max = torch.max(sim_pos, dim=1)
                easy_pos = [torch.index_select(pos[i], 0, sim_pos_max[1][i]) for i in range(len(sim_pos_max[1]))]
                easy_pos = torch.stack(easy_pos, dim=0).squeeze(1)
                score_pos = torch.bmm(query.unsqueeze(dim=1), easy_pos.unsqueeze(dim=-1)).squeeze(dim=-1)  # N*1
        else:  # batch_size: 8
            score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)  # N*1
        score_neg = torch.mm(query, self.memory_queue.t().contiguous())

        logits = torch.cat([score_pos, score_neg], dim=1)  # logits: Nx(1+K)
        # apply temperature
        logits /= self.t
        # labels: positive key indicators
        labels = torch.ones_like(logits)  # BCE

        score = get_score(score_pos, score_neg)
        key_mix = self.anchor_mix_pos(query, pos, weight=self.weight).view(-1, self.dim_in).detach()
        self.memory_queue = torch.cat((self.memory_queue, key), dim=0)[key_mix.size(0):]
        return loss, logits, labels, score
