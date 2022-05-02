from utils import load, dump, del_duplicate, get_config, extractid
import copy
import json
import random

def searchEnt(keyword, entityDic):
    for key, value in entityDic.items():
        if keyword in key:
            return value
    return 'empty'

# 多跳实验
def two(list1, list2):
    tmp_list = []
    if len(list1) > 1 and str(type(list1[-1])) != '<class \'dict\'>':
        s1, r1, o1 = list1[-1]
        tmp_list.extend(list1)
    else:
        s1, r1, o1 = list1
        tmp_list.append(list1)

    s2, r2, o2 = list2
    if o1 == s2:
        tmp_list.append(list2)
    return tmp_list


def find_tpl(list_of_list):
    res_lists = []
    for tmp_list1 in list_of_list:
        list1 = tmp_list1
        for tmp_list2 in list_of_list:
            if tmp_list2 == list1:
                continue
            list2 = tmp_list2
            two_res_list = two(list1, list2)
            if len(two_res_list) > 1:
                list1 = two_res_list
        if list1 != tmp_list1:
            res_lists.append(list1)
    return res_lists


def multi_hop(line, multi_samples, multi_hop_tpls):
    """
    多跳实验：
    法一：从相同KG中提取存在与路径相同的relation
    假定路径(Qid rel1 node1)(node1 rel2 answer)
    1.根据上述的路径，找出类似三元组(s1, rel3, o1)(s2, rel4, o2)
      其中满足: o1 == s2
    2.找到满足第一个要求的节点后，判断relation是否一致
      满足条件: rel3 == rel1 and rel4 == rel2
    3.生成问题，将Qid <--> s1
                answer <--> o2
    """
    new_line = copy.deepcopy(line)
    random.shuffle(new_line['subgraph']['tuples'])
    random.shuffle(new_line['subgraph']['entities'])
    question = new_line['question']
    new_sample = []
    for Qid in multi_samples:
        multi_question = Qid['question'].lower()
        question_text = list(Qid['id2info'].items())[0][1].lower()  # question描述
        # if multi_question == question:
        relations = []
        entities = set()
        for paths in Qid['path']:
            relation = []
            for path in paths:
                if str(type(path)) == '<class \'dict\'>':
                    continue
                for tpl in path:
                    s, r, o = tpl
                    r_ = extractid(r)
                    relation.append(r_)
                    s_, o_ = extractid(s), extractid(o)
                    entities.add(s_)
                    entities.add(o_)
                relations.append(relation)  # 保存多跳路径中三元组的关系集

        new_questions = []  # 新问题
        multi_question_ = copy.deepcopy(multi_question)
        for tpls in multi_hop_tpls:
            KG_relations = []
            new_entities = set()
            new_entity = tpls[0][0]  # 新问题问题实体
            new_entity_ = extractid(new_entity)
            new_answer = copy.deepcopy(tpls[-1][-1])  # 新问题答案
            new_answer_ = extractid(new_answer)
            for tpl_ in tpls:
                s1, r1, o1 = tpl_
                KG_relation = extractid(r1)
                KG_relations.append(KG_relation)  # KG中多跳路径三元组关系集
                new_entities.add(extractid(s1))
                new_entities.add(extractid(o1))
            if KG_relations in relations:  # 两者匹配，若相等，则生成新数据
                new_question_entity = searchEnt(new_entity_, entityDic).lower()  # 找出实体描述
                new_question_answer = searchEnt(new_answer_, entityDic)

                if new_question_entity != 'empty' and new_question_answer != 'empty':
                    new_answer['text'] = new_question_answer
                    new_question = multi_question_.replace(question_text, new_question_entity)
                    new_line['question'] = new_question
                    # print('old_question: ', multi_question)
                    # print('新生成问题：', new_line['question'])
                    new_line['entities'] = [new_entity]
                    if new_question not in new_questions:
                        new_line['answers'] = [new_answer]
                        new_questions.append(new_question)
                    elif new_answer not in new_line['answers']:
                        new_line['answers'].append(new_answer)

        if new_line['question'] != question:
            new_sample.append(new_line)
            break

    return new_sample

with open('datasets/template_data/id2entity.json') as f:
    entityDic = json.load(f)
    f.close()

def main(data_path, multi_samples):
    data_path = data_path + 'train.json'
    data = load(data_path)
    multi_samples = load(multi_samples)
    new_samples = []  # 新生成样本


    for line in data:
        all_triples = line['subgraph']['tuples']
        multi_hop_tpls = find_tpl(all_triples)  # 存在多跳路径的三元组

        # print('原始问题:', line['question'])

        new_sample = multi_hop(line, multi_samples, multi_hop_tpls)
        if len(new_sample) != 0:
            new_samples.extend(new_sample)

    return new_samples

if __name__ == '__main__':
    cfg = get_config()
    mode = cfg['mode']
    multi_samples = cfg['template_folder'] + cfg['template_data']  # datasets/template_data/train_multi_hop.json
    save_path = cfg['save_path']  # datasets/new_samples/kb_03/
    new_samples = main(data_path=cfg['data_folder'], multi_samples=multi_samples)
    dump(save_path + '{}_new_samples.json'.format(mode), new_samples)