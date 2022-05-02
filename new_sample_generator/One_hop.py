from utils import load, dump, extractid
import copy
import random

def one_hop(line, single_samples, new_questions, line_ids):
    """
    单跳实验：
    1、从原始数据集的问题中抽取实体，并查询出其路径和relation
    2、从新生成问题中找到与上述relation相同的数据
    3、生成新的样本数据添加进原始数据集中
    """
    new_line = copy.deepcopy(line)
    random.shuffle(new_line['subgraph']['tuples'])
    random.shuffle(new_line['subgraph']['entities'])
    new_sample = []
    for Qid in single_samples:
        Qid_question = Qid['question'].lower()
        if Qid_question == new_line['question']:  # 根据问题判断出路径和relation
            for path in Qid['path']:
                s, r, o = path
                relation = extractid(r)  # 提取relation
                for new_question in new_questions:
                    for ent in new_question['answer']:
                        answer2id = ent['kb_id']
                    for ent in new_question['entities']:
                        ent2id = ent['kb_id']
                    # 若新生成问题的relation与路径的relation相同，生成新的样本数据
                    # 并判断KG中是否存在这些节点
                    if relation == new_question['relation'] and answer2id in line_ids and ent2id in line_ids:
                        # print('新问题', new_question['question'])
                        new_line['question'] = new_question['question'].lower()  # 新问题
                        new_line['answers'] = new_question['answer']  # 新答案
                        new_line['entities'] = new_question['entities']  # 新问题实体
                        new_sample.append(new_line)
                        break
    return new_sample

if __name__ == '__main__':
    # 加载数据
    mode = 'train'
    file = 'datasets/webqsp/kb_03/'
    data_path = file + '{}.json'.format(mode)
    single_samples_path = 'datasets/template_data/{}_single_hop.json'.format(mode)  # 路径
    new_questions_path = 'datasets/template_data/{}_questions.json'.format(mode)  # 新问题

    data = load(data_path)
    single_samples = load(single_samples_path)
    new_questions = load(new_questions_path)
    new_samples = []
    for line in data:
        # print('初始问题：', line['question'])
        all_triples = line['subgraph']['tuples']
        data_ids = []  # KG中三元组的ent2id集合
        for tpl in all_triples:
            s1, r1, o1 = tpl
            sid, oid = s1['kb_id'], o1['kb_id']
            ids = [sid, oid]
            data_ids.extend(ids)

        new_sample = one_hop(line, single_samples, new_questions, data_ids)
        if len(new_sample) != 0:
            new_samples.append(new_sample)

    dump('datasets/new_samples/single_{}.json'.format(mode), new_samples)



