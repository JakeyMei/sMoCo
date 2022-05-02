import random
import copy
import pandas as pd
import numpy as np

from tqdm import tqdm
from utils import load, dump, get_config, del_duplicate
from One_hop import one_hop
from Multi_hop import find_tpl, multi_hop
from question_generation import question_generation


def extractionQ(samples):
    questions = []
    for Qid in samples:
        Qid_question = Qid['question'].lower()
        questions.append(Qid_question)
    questions = del_duplicate(questions)
    return questions


def rank_data(cfg):

    mode = cfg['mode']
    batch_size = cfg['batch_size']
    data = load(cfg['data_folder'] + cfg['train_data'])
    # multi_samples = load(cfg['template_folder'] + '{}_multi_hop.json'.format(mode))  # 多跳问题模板
    # single_samples = load(cfg['template_folder'] + '{}_single_hop.json'.format(mode))
    # single_new_questions = load(cfg['template_folder'] + '{}_questions.json'.format(mode))  # 单跳新问题模板
    random_question = pd.read_csv(cfg['data_folder'] + '{}_template.csv'.format(mode), usecols=['type', 'description', 'relation'])
    random_question = np.array(random_question)  # 随机问题模板
    new_data = []  # 新训练数据（[1-4：原始训练数据]， [5-batch_size：新样本数据]）
    # eps = cfg['eps'] * 0.1
    # count_multi_hop = 0
    # count_one_hop = 0
    # count_sub_kg = 0
    positive_num = cfg['positive_num']  # 生成新样本个数

    for batch_id in tqdm(range(0, len(data), batch_size)):
        batch = data[batch_id:batch_id + batch_size]
        new_data.extend(batch)
        batch_size = len(batch)
        for line in batch:
            for i in range(positive_num):
                all_triples = line['subgraph']['tuples']
                random.shuffle(all_triples)
                random.shuffle(line['subgraph']['entities'])
                # multi_hop_tpls = find_tpl(all_triples)  # 存在多跳路径的三元组
                # line_ids = []  # KG中三元组的ent2id集合
                relevant_ents = set()  # 答案 + 问题的集合
                # for tpl in all_triples:
                #     s1, r1, o1 = tpl
                #     sid, oid = s1['kb_id'], o1['kb_id']
                #     ids = [sid, oid]
                #     line_ids.extend(ids)
                for ent in line['answers']:
                    relevant_ents.add(ent['kb_id'])
                for ent in line['entities']:
                    relevant_ents.add(ent['text'])

                # # 多跳样本
                # multi_sample = multi_hop(line, multi_samples, multi_hop_tpls)
                seed = random.random()
                # if len(multi_sample) != 0 and seed > eps:
                #     # print('starting multi_hop...')
                #     new_data.append(multi_sample[0])
                #     count_multi_hop = count_multi_hop + 1
                #     continue
                #
                # # 单跳样本
                # single_sample = one_hop(line, single_samples, single_new_questions, line_ids)
                # if len(single_sample) != 0:
                #     # print('starting one_hop...')
                #     new_data.append(single_sample[0])
                #     count_one_hop = count_one_hop + 1
                #     continue

                # 随机样本
                random_sample = []
                new_line = copy.deepcopy(line)
                new_question = question_generation(random_question, all_triples, answer=new_line['answers'])
                if len(new_question) != 0:
                    new_question = random.choice(new_question)
                    new_line['question'] = new_question['question'].lower()  # 新问题
                    new_line['answers'] = new_question['answer']  # 新答案
                    new_line['entities'] = new_question['entities']  # 新问题实体
                    random_sample.append(new_line)
                if len(random_sample) != 0:
                    # print('start random_question...')
                    new_data.append(random_sample[0])
                    # count_one_hop = count_one_hop + 1
                    continue

                # 子集样本
                entities = copy.deepcopy(line['subgraph']['entities'])
                tpls = copy.deepcopy(all_triples)
                KGsubset_sample = []
                del_entities = []  # 删除的实体（随机抽取）
                for j, ent in enumerate(entities):
                    if ent['text'] not in relevant_ents:
                        del_entities.append(ent)
                if len(del_entities) > 1:
                    del_entities = random.choice(del_entities)
                    for tpl_ in tpls:
                        s, r, o = tpl_
                        if s['text'] in del_entities or o['text'] in del_entities:
                            del tpl_
                            break
                new_line_ = copy.deepcopy(line)
                new_line_['subgraph']['entities'] = entities
                new_line_['subgraph']['tuples'] = tpls
                KGsubset_sample.append(new_line_)
                if len(KGsubset_sample) != 0:
                    # print('start KG subset...')
                    new_data.append(KGsubset_sample[0])
                    # count_sub_kg = count_sub_kg + 1
                    continue

    # print('count_multi_hop: ', count_multi_hop)
    # print('count_one_hop: ', count_one_hop)
    # print('count_sub_kg: ', count_sub_kg)
    return new_data

if __name__ == '__main__':
    cfg = get_config()
    new_data = rank_data(cfg)
    dump(cfg['save_path'] + 'new_data_answer.json', new_data)

