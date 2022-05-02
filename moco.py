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


class MoCo(nn.Module):

    def __init__(self, encoder_q, encoder_k, dim_in, K=2000, t=0.07, momentum=0.99):
        super(MoCo, self).__init__()
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dim_in = dim_in
        self.K = K  # 所有的负样本数
        self.t = t  # 常数
        self.momentum = momentum  # 更新参数
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
        # query = query.to(self.device)
        if normalize:
            query = F.normalize(query, dim=1)

        # momentum update
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            _, _, _, key = self.encoder_k(x_k[0], x_k[1])
            idx = torch.randperm(key.size(0))
            # key = key.to(self.device)
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
        # score_neg = torch.topk(score_neg, 8, 1)[0]

        logits = torch.cat([score_pos, score_neg], dim=1)  # logits: Nx(1+K)
        # apply temperature
        logits /= self.t
        # labels: positive key indicators
        labels = torch.ones_like(logits)  # BCE

        score = get_score(score_pos, score_neg)
        # self.memory_queue = torch.cat((self.memory_queue, key), dim=0)[key.size(0):]
        key_mix = self.anchor_mix_pos(query, pos, weight=self.weight).view(-1, self.dim_in)
        self.memory_queue = torch.cat((self.memory_queue, key), dim=0)[key_mix.size(0):]
        return loss, logits, labels, score


class MoCoFT(MoCo):

    def __init__(self, encoder_q, encoder_k, dim_in, K=2000, t=0.07, momentum=0.99,
                 mix_target='pos', postmix_norm=False, expolation_mask=False,
                 dim_mask='both', mask_distribution='uniform', alpha=2.0, norm_target='pos',
                 pos_alpha=2.0, neg_alpha=1.6, sep_alpha=False, mix_jig=False):
        super(MoCoFT, self).__init__(encoder_q, encoder_k, dim_in)

        self.K = K  # 所有的负样本数
        self.t = t  # 常数
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.momentum = momentum
        self.index = 0
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create memory queue
        self.register_buffer('memory', torch.randn(self.K, dim_in))
        self.memory = F.normalize(self.memory, dim=-1)

        assert mix_target in ['pos', 'neg', 'posneg']
        self.mix_target = mix_target
        self.postmix_norm = postmix_norm
        self.expolation_mask = expolation_mask

        assert mask_distribution in ['uniform', 'beta']
        self.mask_distribution = mask_distribution
        assert dim_mask in ['pos', 'neg', 'both', 'none']
        self.dim_mask = dim_mask

        self.alpha = alpha
        self.pos_alpha = pos_alpha
        self.neg_alpha = neg_alpha
        self.sep_alpha = sep_alpha

        if self.expolation_mask:
            assert self.mix_target in ['pos', 'posneg']

        self.mix_jig = mix_jig

        self.norm_target = 'pos'

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue):
        """
        Args:
          k: key feature
          queue: memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def _compute_logit(self, x_q, x_k, queue):

        score_pos = torch.bmm(x_q.unsqueeze(dim=1), x_k.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(x_q, queue.t().contiguous())
        logits = torch.cat([score_pos, score_neg], dim=1)  # logits: Nx(1+K)
        # apply temperature
        logits /= self.t
        # labels: positive key indicators
        labels = torch.ones_like(logits)

        score = get_score(score_pos, score_neg)

        return logits, labels, score

    def forward(self, q, k, normalize=True, all_k=None, mix_now=True, use_hard=False):
        loss_q, _, _, q = self.encoder_q(q[0], q[1])
        q = q.to(self.device)
        batch_size = q.size(0)
        if normalize:
            q = F.normalize(q, dim=1)

        # momentum update
        with torch.no_grad():
            self._momentum_update_key_encoder()
            loss_k, _, _, k = self.encoder_k(k[0], k[1])
            idx = torch.randperm(k.size(0))
            k = k.to(self.device)
            k = k[torch.argsort(idx)]
            if normalize:
                k = F.normalize(k, dim=1)

        if len(k) > 4:
            pos = torch.stack(torch.split(k, 3, dim=0), dim=0)
            sim_pos = self.get_sim(q, pos, flag=1)
            sim_pos_min = torch.min(sim_pos, dim=1)
            hard_pos = [torch.index_select(pos[i], 0, sim_pos_min[1][i]) for i in range(len(sim_pos_min[1]))]
            k = torch.stack(hard_pos, dim=0).squeeze(1)

        # compute logit
        queue = self.memory.clone()

        """
            Mixing targets
        """
        if mix_now:
            if self.mix_target == 'pos':
                mask_shape = q.shape
                if self.mask_distribution == 'uniform':
                    mask = torch.rand(size=mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    mask = np.random.beta(self.alpha, self.alpha, size=mask_shape)

                if self.expolation_mask:
                    mask += 1
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float().cuda()

                q_mix = mask * q + (1 - mask) * k
                k_mix = mask * k + (1 - mask) * q
                q, k = q_mix, k_mix

            elif self.mix_target == 'posneg':
                pos_mask_shape = q.shape
                neg_mask_shape = queue.shape
                if self.mask_distribution == 'uniform':
                    pos_mask = torch.rand(size=pos_mask_shape).cuda()
                    neg_mask = torch.rand(size=neg_mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    pos_mask = np.random.beta(self.alpha, self.alpha, size=pos_mask_shape)
                    neg_mask = np.random.beta(self.alpha, self.alpha, size=neg_mask_shape)

                    if self.sep_alpha:
                        pos_mask = np.random.beta(self.pos_alpha, self.pos_alpha, size=pos_mask_shape)
                        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha, size=neg_mask_shape)

                        if self.dim_mask == 'none':
                            pos_mask = np.random.beta(self.pos_alpha, self.pos_alpha)
                            neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha)
                        elif self.dim_mask == 'pos':
                            neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha)
                        elif self.dim_mask == 'neg':
                            pos_mask = np.random.beta(self.pos_alpha, self.pos_alpha)
                        elif self.dim_mask == 'both':
                            pass

                if self.expolation_mask:
                    pos_mask += 1
                if isinstance(pos_mask, np.ndarray):
                    pos_mask = torch.from_numpy(pos_mask).float().cuda()
                if isinstance(neg_mask, np.ndarray):
                    neg_mask = torch.from_numpy(neg_mask).float().cuda()
                q_mix = pos_mask * q + (1 - pos_mask) * k
                k_mix = pos_mask * k + (1 - pos_mask) * q
                q, k = q_mix, k_mix

                indices = torch.randperm(queue.shape[0]).cuda()
                queue = neg_mask * queue + (1 - neg_mask) * queue[indices]

            else:
                mask_shape = queue.shape
                if self.mask_distribution == 'uniform':
                    mask = torch.rand(size=mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    mask = np.random.beta(self.alpha, self.alpha, size=mask_shape)

                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float().cuda()

                indices = torch.randperm(queue.shape[0]).cuda()
                queue = mask * queue + (1 - mask) * queue[indices]

            if self.postmix_norm:
                if self.norm_target == 'pos':
                    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
                elif self.norm_target == 'neg':
                    queue = F.normalize(queue, dim=1)
                else:
                    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
                    queue = F.normalize(queue, dim=1)
        else:
            print('not mixing')

        logits, labels, score = self._compute_logit(q, k, queue)
        # update memory
        all_k = all_k if all_k is not None else k
        all_k = all_k.float()
        # print(f'self.memory.dtype: {self.memory.dtype}')
        # print(f'all_k.dtype: {all_k.dtype}')
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))

        return loss_q, logits, labels, score

class MoCoV3(MoCo):

    def __init__(self, encoder_q, encoder_k, dim_in, t=0.07, momentum=0.99, mlp_dim=256):
        super(MoCoV3, self).__init__(encoder_q, encoder_k, dim_in, t=t, momentum=momentum)

        self.predictor = nn.Sequential(nn.Linear(self.dim_in, mlp_dim, bias=False),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(mlp_dim, self.dim_in))  # output layer

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.t
        labels = torch.ones_like(logits)
        return nn.BCEWithLogitsLoss()(logits, labels) * (2 * self.t)

    def forward(self, x1, x2, batch_size=16, normalize=False, mode='easy'):
        loss, _, _, q1 = self.encoder_q(x1[0], x1[1], mlp=True)
        _, _, _, q2 = self.encoder_q(x2[0], x2[1], mlp=True)
        q1, q2 = q1.to(self.device), q2.to(self.device)
        q1, q2 = self.predictor(q1), self.predictor(q2)

        # momentum update
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            _, _, _, k1 = self.encoder_k(x1[0], x1[1], mlp=True)
            _, _, _, k2 = self.encoder_k(x2[0], x2[1], mlp=True)
            k1, k2 = k1.to(self.device), k2.to(self.device)

        return loss, self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)