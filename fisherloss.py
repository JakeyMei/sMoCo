import torch
import random
import torch.nn as nn
from torch.nn import functional as F
from moco import get_score

class Fisher_loss(nn.Module):

    def __init__(self, dim_in, encoder_q=None, encoder_k=None, K=2848, t=0.07, momentum=0.5, mode='inbatch', margin_in_loss=0.25):
        super(Fisher_loss, self).__init__()
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dim_in = dim_in
        self.margin_in_loss = margin_in_loss
        self.mode = mode
        self.encoder_q = encoder_q
        if self.mode != 'inbatch':
            self.K = K
            self.t = t
            self.momentum = momentum
            self.encoder_q = encoder_q
            self.encoder_k = encoder_k
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            self.memory_queue = F.normalize(torch.randn(self.K, self.dim_in).to(self.device), dim=-1)

        self.w = F.normalize(torch.randn(self.dim_in, 100).to(self.device), dim=-1)

    def get_negative(self, features, index, batch_size, num=6):
        idx = batch_size//2
        if num == 6:
            f1 = features[0: index]
            f2 = features[index+1: index+idx]
            f3 = features[index+idx+1:]
            return torch.cat((f1, f2, f3), dim=0)
        elif num == 3:
            f1 = features[idx: index+idx]
            f2 = features[index+idx+1:]
            return torch.cat((f1, f2), dim=0)

    def get_similarity_inbatch(self, anchor, neg, neg_num=6):
        similarities = []
        for i in range(anchor.size(0)):
            similarity = [torch.cosine_similarity(anchor[i], neg[i][j], dim=0) for j in range(neg_num)]
            similarity = torch.stack(similarity, 0)
            similarities.append(similarity)
        similarities = torch.stack(similarities, 0)
        return similarities

    def get_similarity_moco(self, anchor, neg):
        similarities = []
        for i in range(anchor.size(0)):
            similarity = [torch.cosine_similarity(anchor[i], neg[j], dim=0) for j in range(neg.size(0))]
            similarity = torch.stack(similarity, 0)
            similarities.append(similarity)
        similarities = torch.stack(similarities, 0)
        return similarities

    def get_logits(self, anchor, pos, neg):
        score_pos = torch.bmm(anchor.unsqueeze(dim=1), pos.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(anchor, neg.t().contiguous())
        return score_pos, score_neg

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def loss_FDA(self, o1, o2, o3, lambda_=0.01, contrastive=False):
        epsilon_Sw, epsilon_Sb = 0.0001, 0.0001

        # calculation of within scatter:
        temp1 = o1 - o2  # b * q
        S_within = torch.matmul(temp1.t(), temp1)  # (q * b) * (b * q) = (q * q)

        # calculation of within scatter:
        temp2 = o1 - o3
        S_between = torch.matmul(temp2.t(), temp2)

        # strengthen main diagonal of S_within and S_between:
        I_matrix = torch.eye(S_within.shape[0], dtype=torch.uint8).float().to(self.device)
        I_matrix_Sw = I_matrix * epsilon_Sw
        I_matrix_Sb = I_matrix * epsilon_Sb
        S_within = torch.add(S_within, I_matrix_Sw)
        S_between = torch.add(S_between, I_matrix_Sb)

        # calculation of variance of projection considering within scatter:
        temp3 = torch.matmul(self.w.t(), S_within)
        temp3 = torch.matmul(temp3, self.w)
        within_scatter_term = torch.trace(temp3)

        # calculation of variance of projection considering between scatter:
        temp4 = torch.matmul(self.w.t(), S_between)
        temp4 = torch.matmul(temp4, self.w)
        between_scatter_term = torch.trace(temp4)

        # calculation of loss:
        if contrastive:
            y = random.uniform(0, 1)
            if y <= 0.5:
                loss_ = (2-lambda_) * within_scatter_term
            else:
                loss_neg = self.margin_in_loss - (lambda_ * between_scatter_term)
                zero = torch.zeros_like(loss_neg)
                loss_ = torch.where(loss_neg < 0., zero, loss_neg)
            loss = loss_
        else:
            loss_ = self.margin_in_loss + ((2 - lambda_) * within_scatter_term) - (lambda_ * between_scatter_term)
            zero = torch.zeros_like(loss_)
            loss = torch.where(loss_ < 0., zero, loss_)

        return loss

    def loss_contrastive(self, o1, o2, o3):
        d_pos = torch.sum(torch.pow((o1 - o2), 2), dim=1)
        d_neg = torch.sum(torch.pow((o1 - o3), 2), dim=1)

        y = random.uniform(0, 1)
        if y <= 0.5:
            loss = d_pos
        else:
            loss_ = self.margin_in_loss - d_neg
            zero = torch.zeros_like(loss_)
            loss = torch.where(loss_ < 0., zero, loss_)

        loss = torch.mean(loss)

        return loss

    def loss_triplet(self, o1, o2, o3):
        d_pos = torch.sum(torch.pow((o1 - o2), 2), dim=1)
        d_neg = torch.sum(torch.pow((o1 - o3), 2), dim=1)

        loss_ = self.margin_in_loss + d_pos - d_neg
        zero = torch.zeros_like(loss_)
        loss = torch.where(loss_ < 0., zero, loss_)
        loss = torch.mean(loss)

        return loss

    def forward(self, loss_type, feed_q=None, feed_k=None, neg_num=6):

        if self.mode == 'inbatch':
            loss_main, _, _, features = self.encoder_q(feed_q)
            features = features.to(self.device)
            batch_size = features.size(0)
            idx = batch_size // 2
            o1 = features[torch.arange(batch_size) < idx]  # anchor
            o2 = features[torch.arange(batch_size) >= idx]  # positive
            if batch_size != 2:
                num = (batch_size - 2) if neg_num == 6 else (idx - 1)
                o3 = [self.get_negative(features, i, batch_size) for i in range(idx)]
                similarity = self.get_similarity_inbatch(o1, o3, neg_num=num)
                max_idx = torch.max(similarity, dim=1)[1]
                o3 = [torch.index_select(o3[i], 0, max_idx[i]) for i in range(max_idx.size(0))]
                o3 = torch.stack(o3).squeeze(1)
            else:
                o3 = torch.zeros_like(o1)
        else:
            with torch.no_grad():
                self._momentum_update_key_encoder()

            loss_main, _, _, o1 = self.encoder_q(feed_q[0], feed_k[1])
            _, _, _, o2 = self.encoder_k(feed_k[0], feed_k[1])
            o1 = o1.to(self.device)
            o2 = o2.to(self.device)
            similarity = self.get_similarity_moco(o1, self.memory_queue)
            max_idx = torch.max(similarity, dim=1)[1]
            o3 = [self.memory_queue[i] for i in max_idx.tolist()]
            o3 = torch.stack(o3).squeeze(1)
            self.memory_queue = torch.cat((self.memory_queue, o2), dim=0)[o2.size(0):]

        if loss_type == 'triplet':
            loss = self.loss_triplet(o1, o2, o3)
        elif loss_type == 'contrastive':
            loss = self.loss_contrastive(o1, o2, o3)
        elif loss_type == 'FDA':
            self.w = self.encoder_q.get_weights(self.w).data
            loss = self.loss_FDA(o1, o2, o3)
        elif loss_type == 'FDA_contrastive':
            self.w = self.encoder_q.get_weights(self.w).data
            loss = self.loss_FDA(o1, o2, o3, contrastive=True)
        else:
            loss = 0.

        score_pos, score_neg = self.get_logits(o1, o2, o3)
        score = get_score(score_pos, score_neg)


        return loss_main, loss, score


if __name__ == '__main__':
    features = torch.rand(6, 100).cuda()
    criterion = Fisher_loss(100).cuda()
    for i in range(20):
        loss, score = criterion('FDA', features, 6)
        print(loss)
        print(score.shape)