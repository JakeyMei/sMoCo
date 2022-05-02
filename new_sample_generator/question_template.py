import pandas as pd
from utils import load, del_duplicate
import re

mode = 'train'
data_file = 'kb_05'
filename = 'datasets/webqsp/{}/{}.json'.format(data_file, mode)
data = load(filename)

templates = []  # 问题模板
for line in data:
    ent2ids = []  # 存放答案中的实体ID

    question = line['question']
    print(question)
    types = ['what', 'where', 'when', 'who', 'which', 'how', 'why']
    for type_ in types:
        patt = re.compile(r'{}'.format(type_))  # 提取问题类型：what where when who which how why
        Q_type = patt.findall(question)
        print(Q_type)
        if Q_type:
            Q_type = Q_type[0]
            break
    answer = line['answers']
    all_triples = line['subgraph']['tuples']

    for ent in answer:
        ent2id = ent['kb_id']
        for tpl in all_triples:
            s, r, o = tpl
            if ent2id in s['kb_id'] or ent2id in o['kb_id']:
                relation = r['text'].strip('<>')[3:]
                # print(Q_type, '\t', question, '\t', relation)
                template_ = [Q_type, question, relation]
                templates.append(template_)

templates = del_duplicate(templates)  # 去重

columns = ['type', 'description', 'relation']  # 数据有三列，列名分别为type, description, relation

test = pd.DataFrame(columns=columns, data=templates)
test.to_csv('datasets/template_data/{}/{}_template.csv'.format(data_file, mode), encoding='utf-8')