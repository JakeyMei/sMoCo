# coding=UTF-8
import torch
import numpy as np
import random
import operator
import json

from torch import nn
from data_generator import DataLoader
from model import KAReader
from simclr import Simclr
from util import get_config, cal_accuracy, load_documents, get_q_k
from moco import MoCo, MoCoFT, MoCoV3
from simsiam import SimSiam
from fisherloss import Fisher_loss

from tensorboardX import SummaryWriter


def f1_and_hits(answers, candidate2prob, eps):
    """
    answers:真正的答案
    candidate2prob:每个候选集对应的概率
    eps:召回率阈值 0.05
    """
    retrieved = []  # 候选集
    correct = 0
    for c, prob in candidate2prob.items():
        if prob > eps:
            retrieved.append(c)
            if c in answers:
                correct += 1
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0
        else:
            return 0.0, 1.0
    else:
        best_ans = get_best_ans(candidate2prob)
        hits = float(best_ans in answers)  # 若预测答案与正确答案一致，返回1.0，否则返回0.0
        if len(retrieved) == 0:
            return 0.0, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return f1, hits


# 最高概率
def get_best_ans(candidate2prob):
    best_ans, max_prob = -1, 0
    for c, prob in candidate2prob.items():
        if prob > max_prob:
            max_prob = prob
            best_ans = c
    return best_ans


class train():
    def __init__(self, cfg):

        self.use_doc = cfg['use_doc']
        self.weight = cfg['weight']
        self.use_moco = cfg['use_moco']
        self.use_fisher = cfg['use_fisher']
        self.fisher_loss = cfg['fisher_loss']  # triplet, contrastive, FDA, FDA_contrastive
        self.entity_dim = cfg['entity_dim']
        self.train_method = cfg['train_choice']
        self.num_epochs = cfg['num_epoch']
        self.batch_size = cfg['batch_size']
        self.K = cfg['queue_nums']
        self.flag = 2 if self.batch_size == 8 else 4

        self.tf_logger = SummaryWriter('tf_logs/' + cfg['model_id'])
        self.documents = load_documents(cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])  # train and test share the same set of documents

        # train data
        self.train_data = DataLoader(cfg, self.documents)
        self.valid_data = DataLoader(cfg, self.documents, mode='test')

        # model
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.base_model = KAReader(cfg).to(self.device)
        self.encoder_q = KAReader(cfg).to(self.device)
        self.encoder_k = KAReader(cfg).to(self.device)
        self.inbatch = Simclr().to(self.device)
        self.dim_in = self.entity_dim * 4 if self.use_doc else self.entity_dim * 2
        # self.dim_in = self.train_data.max_local_entity
        self.base_moco = MoCo(self.encoder_q, self.encoder_k, self.dim_in, K=self.K, momentum=cfg['momentum']).to(self.device)
        self.moco_mix = MoCoFT(self.encoder_q, self.encoder_k, self.dim_in, K=self.K, mix_target=cfg['mixnorm_target'],
                                             postmix_norm=cfg['postmix_norm'],
                                             expolation_mask=cfg['expolation_mask'], dim_mask=cfg['dim_mask'],
                                             mask_distribution=cfg['mask_distribution'], alpha=cfg['beta_alpha'],
                                             pos_alpha=cfg['pos_alpha'], neg_alpha=cfg['neg_alpha'],
                                             sep_alpha=cfg['sep_alpha']).to(self.device)
        self.mocov3 = MoCoV3(self.encoder_q, self.encoder_k, self.dim_in, momentum=cfg['momentum']).to(self.device)
        self.simsiam = SimSiam(self.base_model, self.dim_in).to(self.device)

        self.loss_bce = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_simsiam = nn.CosineSimilarity(dim=1).to(self.device)

        if self.use_moco:
            self.model = self.encoder_q
            self.fisherloss = Fisher_loss(self.dim_in, encoder_q=self.model, encoder_k=self.encoder_k, mode='moco').to(self.device)
            if cfg['moco_mode'] == 'MoCoFT':
                self.moco = self.moco_mix
                self.cl_mode = 'MoCoFT'
            elif cfg['moco_mode'] == 'MoCoV3':
                self.moco = self.mocov3
                self.cl_mode = 'MoCoV3'
            elif cfg['moco_mode'] == 'SimSiam':
                self.moco = self.simsiam
                self.cl_mode = 'SimSiam'
            else:
                self.moco = self.base_moco
                self.cl_mode = 'moco' if self.batch_size == 8 else 'moco_{}'.format(cfg['pos_mode'])
        else:
            self.model = self.base_model
            self.fisherloss = Fisher_loss(self.dim_in, encoder_q=self.model, mode='inbatch').to(self.device)
            self.cl_mode = 'inbatch_fisher' if self.use_fisher else 'inbatch'

        self.trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = torch.optim.Adam(self.trainable, lr=cfg['learning_rate'])

        if cfg['lr_schedule']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, [30], gamma=0.5)

        self.model.train()

        # best model & evaluation

        self.file = cfg['data_folder'][16:-1]
        self.normalize = True if self.file != 'full' else False
        self.scores = []  # pos & neg score
        self.total_hits = []
        self.total_f1 = []
        self.train_loss = []
        self.train_loss_cl = []
        self.best_val_f1 = 0
        self.best_val_hits = 0
        self.avg_best = 0
        self.avg_best_f1 = 0
        self.avg_best_hits = 0

        # train_method choice
        assert self.train_method in ['joint', 'alternate', 'serial']
        if self.train_method == 'joint':
            self.train_joint()
        elif self.train_method == 'alternate':
            self.train_alternate()
        elif self.train_method == 'serial':
            self.train_serial()

    def train_joint(self):
        for epoch in range(self.num_epochs):
            batcher = self.train_data.batcher(shuffle=True)  # 选取8条数据
            train_loss = []
            train_loss_cl = []
            scores = []
            for feed in batcher:
                if self.use_moco:
                    feed_q, feed_k = get_q_k(feed, self.batch_size, flag=self.flag)
                    if len(feed_q[1]) == 0:
                        continue
                    if self.use_fisher:
                        loss, loss_cl, score = self.fisherloss(self.fisher_loss, feed_q=feed_q, feed_k=feed_k)
                    elif self.cl_mode == 'MoCoFT':
                        loss, logits, labels, score = self.moco(feed_q, feed_k, normalize=self.normalize)
                        loss_cl = self.loss_bce(logits, labels)
                    elif self.cl_mode == 'MoCoV3':
                        score = []
                        loss, loss_cl = self.moco(feed_q, feed_k, self.batch_size)
                    elif self.cl_mode == 'SimSiam':
                        score = []
                        loss, p1, p2, z1, z2 = self.simsiam(feed_q, feed_k)
                        loss_cl = -(self.loss_simsiam(p1, z2).mean() + self.loss_simsiam(p2, z1).mean()) * 0.5
                    else:
                        loss, logits, labels, score = self.moco(feed_q, feed_k, batch_size=self.batch_size, normalize=self.normalize,
                                                                mode=cfg['pos_mode'])
                        loss_cl = self.loss_bce(logits, labels)
                else:
                    if self.use_fisher:
                        loss, loss_cl, score = self.fisherloss(self.fisher_loss, feed_q=feed)
                    else:
                        loss, pred, pred_dist, features = self.model(feed)
                        logits, labels, score = self.inbatch(features, self.device)
                        loss_cl = self.loss_bce(logits, labels)

                loss = loss + self.weight * loss_cl
                train_loss.append(loss.item())
                train_loss_cl.append(loss_cl)
                # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                self.optim.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, cfg['gradient_clip'])
                self.optim.step()
                scores.append(score)
            self.scores.append(scores)
            self.tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)
            self.save_best_valid(epoch)
            self.total_hits.append(self.best_val_hits)
            self.total_f1.append(self.best_val_f1)
            self.train_loss.append(train_loss)
            self.train_loss_cl.append(train_loss_cl)
        print('save final model')
        self.save_evaluation()
        torch.save(self.model.state_dict(), 'model/{}/{}_final.pt'.format(cfg['name'], cfg['model_id']))

        model_save_path = 'model/{}/{}_best1.pt'.format(cfg['name'], cfg['model_id'])
        self.model.load_state_dict(torch.load(model_save_path))

        print('\n..........Finished training, start testing.......')
        test_data = DataLoader(cfg, self.documents, mode='test')
        # print('finished training, testing final model...')
        self.model.eval()
        test(self.model, test_data, cfg['eps'])

    def train_alternate(self):
        for epoch in range(self.num_epochs):
            w = 1 if (epoch+1) % 5 == 0 else 0
            batcher = self.train_data.batcher(shuffle=True)  # 选取8条数据
            train_loss = []
            train_loss_cl = []
            scores = []
            for feed in batcher:
                if self.use_moco:
                    feed_q, feed_k = get_q_k(feed, self.batch_size, flag=self.flag)
                    if len(feed_q[1]) == 0:
                        continue
                    if self.use_fisher:
                        loss, loss_cl, score = self.fisherloss(self.fisher_loss, feed_q=feed_q, feed_k=feed_k)
                    elif self.cl_mode == 'MoCoFT':
                        loss, logits, labels, score = self.moco(feed_q, feed_k, normalize=self.normalize)
                        loss_cl = self.loss_bce(logits, labels)
                    elif self.cl_mode == 'MoCoV3':
                        score = []
                        loss, loss_cl = self.moco(feed_q, feed_k, self.batch_size)
                    elif self.cl_mode == 'SimSiam':
                        score = []
                        loss, p1, p2, z1, z2 = self.simsiam(feed_q, feed_k)
                        loss_cl = -(self.loss_simsiam(p1, z2).mean() + self.loss_simsiam(p2, z1).mean()) * 0.5
                    else:
                        loss, logits, labels, score = self.moco(feed_q, feed_k, batch_size=self.batch_size,
                                                                normalize=self.normalize,
                                                                mode=cfg['pos_mode'])
                        loss_cl = self.loss_bce(logits, labels)
                else:
                    if self.use_fisher:
                        loss, loss_cl, score = self.fisherloss(self.fisher_loss, feed_q=feed)
                    else:
                        loss, pred, pred_dist, features = self.model(feed)
                        logits, labels, score = self.inbatch(features, self.device)
                        loss_cl = self.loss_bce(logits, labels)
                loss = (1 - w) * loss + w * loss_cl
                train_loss.append(loss.item())
                train_loss_cl.append(loss_cl)
                # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                self.optim.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, cfg['gradient_clip'])
                self.optim.step()
                scores.append(score)
            self.scores.append(scores)
            self.tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)
            self.save_best_valid(epoch)
            self.total_hits.append(self.best_val_hits)
            self.total_f1.append(self.best_val_f1)
            self.train_loss.append(train_loss)
            self.train_loss_cl.append(train_loss_cl)
        print('save final model')
        self.save_evaluation()
        torch.save(self.model.state_dict(), 'model/{}/{}_final.pt'.format(cfg['name'], cfg['model_id']))

        model_save_path = 'model/{}/{}_best2.pt'.format(cfg['name'], cfg['model_id'])
        self.model.load_state_dict(torch.load(model_save_path))

        print('\n..........Finished training, start testing.......')
        test_data = DataLoader(cfg, self.documents, mode='test')
        # print('finished training, testing final model...')
        self.model.eval()
        test(self.model, test_data, cfg['eps'])

    def train_serial(self):
        for epoch in range(self.num_epochs):
            w = 1 if epoch < 20 else 0
            batcher = self.train_data.batcher(shuffle=True)  # 选取8条数据
            train_loss = []
            train_loss_cl = []
            scores = []
            for feed in batcher:
                if self.use_moco:
                    feed_q, feed_k = get_q_k(feed, self.batch_size, flag=self.flag)
                    if len(feed_q[1]) == 0:
                        continue
                    if self.use_fisher:
                        loss, loss_cl, score = self.fisherloss(self.fisher_loss, feed_q=feed_q, feed_k=feed_k)
                    elif self.cl_mode == 'MoCoFT':
                        loss, logits, labels, score = self.moco(feed_q, feed_k, normalize=self.normalize)
                        loss_cl = self.loss_bce(logits, labels)
                    elif self.cl_mode == 'MoCoV3':
                        score = []
                        loss, loss_cl = self.moco(feed_q, feed_k, self.batch_size)
                    elif self.cl_mode == 'SimSiam':
                        score = []
                        loss, p1, p2, z1, z2 = self.simsiam(feed_q, feed_k)
                        loss_cl = -(self.loss_simsiam(p1, z2).mean() + self.loss_simsiam(p2, z1).mean()) * 0.5
                    else:
                        loss, logits, labels, score = self.moco(feed_q, feed_k, batch_size=self.batch_size,
                                                                normalize=self.normalize,
                                                                mode=cfg['pos_mode'])
                        loss_cl = self.loss_bce(logits, labels)
                else:
                    if self.use_fisher:
                        loss, loss_cl, score = self.fisherloss(self.fisher_loss, feed_q=feed)
                    else:
                        loss, pred, pred_dist, features = self.model(feed)
                        logits, labels, score = self.inbatch(features, self.device)
                        loss_cl = self.loss_bce(logits, labels)
                loss = (1 - w) * loss + w * loss_cl
                train_loss.append(loss.item())
                train_loss_cl.append(loss_cl)
                # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                self.optim.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, cfg['gradient_clip'])
                self.optim.step()
                scores.append(score)
            self.scores.append(scores)
            self.tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)
            self.save_best_valid(epoch)
            self.total_hits.append(self.best_val_hits)
            self.total_f1.append(self.best_val_f1)
            self.train_loss.append(train_loss)
            self.train_loss_cl.append(train_loss_cl)
        print('save final model')
        self.save_evaluation()
        torch.save(self.model.state_dict(), 'model/{}/{}_final.pt'.format(cfg['name'], cfg['model_id']))

        model_save_path = 'model/{}/{}_best1.pt'.format(cfg['name'], cfg['model_id'])
        self.model.load_state_dict(torch.load(model_save_path))

        print('\n..........Finished training, start testing.......')
        test_data = DataLoader(cfg, self.documents, mode='test')
        # print('finished training, testing final model...')
        self.model.eval()
        test(self.model, test_data, cfg['eps'])

    def save_best_valid(self, epoch):
        val_f1, val_hits = test(self.model, self.valid_data, cfg['eps'])
        if cfg['lr_schedule']:
            self.scheduler.step()
        self.tf_logger.add_scalar('eval_f1', val_f1, epoch)
        self.tf_logger.add_scalar('eval_hits', val_hits, epoch)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
        if val_hits > self.best_val_hits:
            self.best_val_hits = val_hits
            torch.save(self.model.state_dict(), 'model/{}/{}_best1.pt'.format(cfg['name'], cfg['model_id']))
        val = (val_f1 + val_hits) / 2
        if val > self.avg_best:
            self.avg_best = val
            self.avg_best_f1 = val_f1
            self.avg_best_hits = val_hits
            torch.save(self.model.state_dict(), 'model/{}/{}_best2.pt'.format(cfg['name'], cfg['model_id']))
        print('epoch:', epoch)
        print('evaluation best f1:{} current:{}'.format(self.best_val_f1, val_f1))
        print('evaluation best hits:{} current:{}'.format(self.best_val_hits, val_hits))
        print('evaluation avg_best f1:{} hits:{}'.format(self.avg_best_f1, self.avg_best_hits))

    def save_evaluation(self):
        if self.use_fisher:
            np.save('model/evaluation/{}/scores_{}_{}_{}_fisher'.format(self.file, self.train_method, self.weight,
                                                                 self.cl_mode), np.array(self.scores))
            np.save('model/evaluation/{}/hits_{}_{}_{}_fisher'.format(self.file, self.train_method, self.weight,
                                                               self.cl_mode), np.array(self.total_hits))
            np.save('model/evaluation/{}/f1_{}_{}_{}_fisher'.format(self.file, self.train_method, self.weight,
                                                             self.cl_mode), np.array(self.total_f1))
        else:
            np.save('model/evaluation/{}/scores_{}_{}_{}_{}'.format(self.file, self.train_method, self.weight,
                                                                        self.cl_mode, self.K),  np.array(self.scores))
            np.save('model/evaluation/{}/hits_{}_{}_{}_{}'.format(self.file, self.train_method, self.weight,
                                                                      self.cl_mode, self.K), np.array(self.total_hits))
            np.save('model/evaluation/{}/f1_{}_{}_{}_{}'.format(self.file, self.train_method, self.weight,
                                                                    self.cl_mode, self.K), np.array(self.total_f1))
            np.save('model/evaluation/{}/loss_{}_{}_{}_{}'.format(self.file, self.train_method, self.weight,
                                                             self.cl_mode, self.K), np.array(self.train_loss))
            np.save('model/evaluation/{}/loss_cl_{}_{}_{}_{}'.format(self.file, self.train_method, self.weight,
                                                               self.cl_mode, self.K), np.array(self.train_loss_cl))

def test(model, test_data, eps):
    model.eval()
    batcher = test_data.batcher()
    id2entity = test_data.id2entity
    f1s, hits = [], []
    questions = []
    pred_answers = []
    for feed in batcher:
        _, pred, pred_dist, _ = model(feed, mode='test')
        acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
        batch_size = pred_dist.size(0)
        batch_answers = feed['answers_']
        questions += feed['questions_']
        batch_candidates = feed['candidate_entities']
        pad_ent_id = len(id2entity)
        for batch_id in range(batch_size):
            answers = batch_answers[batch_id]
            candidates = batch_candidates[batch_id, :].tolist()
            probs = pred_dist[batch_id, :].tolist()
            candidate2prob = {}
            for c, p in zip(candidates, probs):
                if c == pad_ent_id:
                    continue
                else:
                    candidate2prob[c] = p
            f1, hit = f1_and_hits(answers, candidate2prob, eps)
            best_ans = get_best_ans(candidate2prob)
            best_ans = id2entity.get(best_ans, '')

            pred_answers.append(best_ans)
            f1s.append(f1)
            hits.append(hit)
    print('evaluation.......')
    print('how many eval samples......', len(f1s))
    print('avg_f1', np.mean(f1s))
    print('avg_hits', np.mean(hits))

    model.train()
    return np.mean(f1s), np.mean(hits)


if __name__ == "__main__":
    # config_file = sys.argv[2]
    cfg = get_config()
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    if cfg['mode'] == 'train':
        train(cfg)
    elif cfg['mode'] == 'test':
        documents = load_documents(cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
        test_data = DataLoader(cfg, documents, mode='test')
        model = KAReader(cfg)
        model = model.to(torch.device('cuda'))
        model_save_path = 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])
        # model_save_path = 'model/{}/{}_final.pt'.format(cfg['name'], cfg['model_id'])
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        test(model, test_data, cfg['eps'])
    else:
        assert False, "--train or --test?"
