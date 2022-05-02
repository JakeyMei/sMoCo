from torch.nn import Module
import torch
import random
from torch.nn import functional as F
from moco import get_score
from util import count_true_answers


class Simclr(Module):
    def __init__(self, batch_size=None, n_views=2, temperature=0.07):
        """
        batch_size: 批数量，这个值可以为空默认为 feature 第一维度的一般
        n_views: 对比学习中正样本 默认只支持一张正样本
        temperature: 超参用来调节损失
        """
        super(Simclr, self).__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature

    def forward(self, features, device):

        features = features.to(device)
        labels = torch.cat([torch.arange(self.batch_size if self.batch_size else features.size(0) // self.n_views) 
                for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.t())
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.uint8).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select and combine multiple positives
        positives = similarity_matrix[labels.type(torch.uint8)].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.type(torch.uint8)].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.ones_like(logits)  # BCE
        logits = logits / self.temperature
        score = get_score(positives, negatives)

        return logits, labels, score


if __name__ == '__main__':
    m = 8
    features = torch.rand(m, 10)  # 刚开始是一个16 * 100
    # print('features:', features)
    criterion = ContrastiveLearningLoss()
    loss_con = criterion(features, 1)
    print(loss_con)