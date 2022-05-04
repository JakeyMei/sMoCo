# coding=UTF-8
import torch
import numpy as np
import random
import operator
import json

from torch import nn
from data_generator import DataLoader
from model import KAReader
from util import get_config, cal_accuracy, load_documents, get_q_k
from smoco import sMoCo

from tensorboardX import SummaryWriter


def f1_and_hits(answers, candidate2prob, eps):
    retrieved = []
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
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 0.0, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return f1, hits


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
        self.use_smoco = cfg['use_smoco']
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
        self.model = KAReader(cfg).to(self.device)            

        self.loss_bce = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_simsiam = nn.CosineSimilarity(dim=1).to(self.device)

        if self.use_smoco:
            self.encoder_k = KAReader(cfg).to(self.device)
            self.dim_in = self.entity_dim * 4 if self.use_doc else self.entity_dim * 2
            self.smoco = sMoCo(self.model, self.encoder_k, self.dim_in, K=self.K, momentum=cfg['momentum']).to(self.device)

        self.trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = torch.optim.Adam(self.trainable, lr=cfg['learning_rate'])

        if cfg['lr_schedule']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, [30], gamma=0.5)

        self.model.train()

        # best model & evaluation

        self.file = cfg['data_folder'][16:-1]
        self.normalize = True if self.file != 'full' else False
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
            batcher = self.train_data.batcher(shuffle=True)
            train_loss = []
            for feed in batcher:
                loss_cl = 0
                if self.use_smoco:
                    feed_q, feed_k = get_q_k(feed, self.batch_size, flag=self.flag)
                    if len(feed_q[1]) == 0:
                        continue                    
                    loss, logits, labels, score = self.smoco(feed_q, feed_k, batch_size=self.batch_size, normalize=self.normalize,
                                                            mode=cfg['pos_mode'])
                    loss_cl = self.loss_bce(logits, labels)
                else:
                    loss, pred, pred_dist, features = self.model(feed)
                loss = loss + self.weight * loss_cl
                train_loss.append(loss.item())
                # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                self.optim.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, cfg['gradient_clip'])
                self.optim.step()
            self.tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)
            self.save_best_valid(epoch)
        print('save final model')
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
            batcher = self.train_data.batcher(shuffle=True)
            train_loss = []
            for feed in batcher:
                loss_cl = 0
                if self.use_smoco:
                    feed_q, feed_k = get_q_k(feed, self.batch_size, flag=self.flag)
                    if len(feed_q[1]) == 0:
                        continue
                    loss, logits, labels, score = self.smoco(feed_q, feed_k, batch_size=self.batch_size,
                                                            normalize=self.normalize,
                                                            mode=cfg['pos_mode'])
                    loss_cl = self.loss_bce(logits, labels)
                else:
                    loss, pred, pred_dist, features = self.model(feed)
                loss = (1 - w) * loss + w * loss_cl
                train_loss.append(loss.item())
                # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                self.optim.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, cfg['gradient_clip'])
                self.optim.step()
            self.tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)
            self.save_best_valid(epoch)
        print('save final model')
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
            batcher = self.train_data.batcher(shuffle=True)
            train_loss = []
            for feed in batcher:
                loss_cl = 0
                if self.use_moco:
                    feed_q, feed_k = get_q_k(feed, self.batch_size, flag=self.flag)
                    if len(feed_q[1]) == 0:
                        continue
                    loss, logits, labels, score = self.smoco(feed_q, feed_k, batch_size=self.batch_size,
                                                            normalize=self.normalize,
                                                            mode=cfg['pos_mode'])
                    loss_cl = self.loss_bce(logits, labels)
                else:
                    loss, pred, pred_dist, features = self.model(feed)
                loss = (1 - w) * loss + w * loss_cl
                train_loss.append(loss.item())
                # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
                # train_acc.append(acc)
                # train_max_acc.append(max_acc)
                self.optim.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, cfg['gradient_clip'])
                self.optim.step()
            self.tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)
            self.save_best_valid(epoch)
        print('save final model')
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
