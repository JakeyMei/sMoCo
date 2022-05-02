import random
import copy
import pandas as pd
import numpy as np
from utils import load, dump, del_duplicate, extractid, get_config

relations_path = 'datasets/template_data/relations.json'
id2entity_path = 'datasets/template_data/id2entity.json'

# 加载数据
relations = load(relations_path, mode='dict')  # 关系描述
id2entity = load(id2entity_path, mode='dict')  # 实体描述
exist_ent = list(id2entity.keys())  # 存在实体信息的实体ID
exist_rel = list(relations.keys())

def question_generation(question_template, all_triples, answer=None):
    """
    1.找寻出存在映射的entity
    2.提取A<-->C:describe of relation
    3.提取Q_type
    4.生成新问题
    """
    new_question_ = []
    tpls = copy.deepcopy(all_triples)
    answers = {}
    for i in answer:
        answers[i['kb_id']] = i['text']
    for tpl in tpls:
        s, r, o = tpl

        # 1.找寻出存在映射的entity
        '''
          1) s: ent2id, o: ent2id
          2) s: ent2id, o: date
          3) s: date,   o: ent2id
          4) s: date,   o: date
        '''
        question_entity = 0
        answer_entity = 0
        if s['kb_id'] in list(answers):
            o_ = extractid(o)
            if o_ in exist_ent:
                question_entity = id2entity[o_]
            else:
                continue
            answer_entity = answers[s['kb_id']]
        elif o['kb_id'] in list(answers):
            s_ = extractid(s)
            if s_ in exist_ent:
                question_entity = id2entity[s_]
            else:
                continue
            answer_entity = answers[o['kb_id']]

        # else:
        #     if s['text'][0] is "<":
        #         s_ = s['text'].strip("<>")[3:]
        #         if s_ in exist_ent:
        #             A = id2entity[s_]
        #             # 1) s: ent2id, o: ent2id
        #             if o['text'][0] is "<":
        #                 o_ = o['text'].strip("<>")[3:]
        #                 if o_ in exist_ent:
        #                     C = id2entity[o_]
        #                 else:
        #                     continue
        #             # 2) s: ent2id, o: date
        #             else:
        #                 C = o['text']
        #         else:
        #             continue
        #     else:
        #         # 3) s: date, o: ent2id
        #         A = s['text']
        #         if o['text'][0] is "<":
        #             o_ = o['text'].strip("<>")[3:]
        #             if o_ in exist_ent:
        #                 C = id2entity[o_]
        #             else:
        #                 continue
        #         # 4) s: date,   o: date
        #         else:
        #             C = o['text']
        if question_entity and answer_entity:
            # 2.提取A<-->C:describe of relation
            r = r['text'].strip("<>")[3:]
            if r in exist_rel:
                des = relations[r]
                # print(A, '<--->', C)

                # 3.提取Q_type：['what', 'where', 'when', 'who', 'which', 'how']
                Qtypes = []
                for template in question_template:
                    if template[2] == r:
                        if template[0] not in Qtypes:
                            Qtypes.append((template[0], r))
                Qtypes = del_duplicate(Qtypes)

                # 4.生成新问题
                random_num = len(Qtypes)
                if len(des) < random_num:
                    random_num = len(des)
                des_random = random.sample(des, random_num)
                # print(des_random)
                for Qtype in Qtypes:
                    for describe in des_random:
                        new_question = {}
                        describe = describe.replace('[X]', Qtype[0])
                        describe = describe.replace('[Y]', question_entity)
                        new_question['question'] = describe
                        o['text'] = answer_entity
                        new_question.setdefault('answer', []).append(o)
                        new_question.setdefault('entities', []).append(s)
                        new_question['relation'] = Qtype[1]
                        new_question_.append(new_question)
                        # print(new_question)
                break
        new_question_ = del_duplicate(new_question_)
    return new_question_


if __name__ == '__main__':
    cfg = get_config()
    mode = cfg['mode']
    templates_path = cfg['data_folder'] + '{}_template.csv'.format(mode)
    filename = cfg['data_folder'] + cfg['train_data']

    templates = pd.read_csv(templates_path, usecols=['type', 'description', 'relation'])
    templates = np.array(templates)  # 问题模板

    data = load(filename)
    found = []  # <A B C>都存在的数据
    notfound = []  # 不满足<A B C>都存在的数据
    new_questions = []  # 新问题
    for line in data:
        print('原始问题：', line['question'])
        all_triples = line['subgraph']['tuples']
        new_question = question_generation(templates, all_triples, answer=line['answers'])

        print(len(new_question))
        print(new_question)
        if len(new_question) != 0:
            new_questions.extend(new_question)
            break
    # 保存文件
    # dump('tmp/notfound_subgraph.json', notfound)  # 不符合要求的subgraph
    # dump('tmp/found_subgraph.json', found)  # 符合要求的subgraph
    # dump('datasets/template_data/{}_questions.json'.format(mode), new_questions)  # 生成的新问题