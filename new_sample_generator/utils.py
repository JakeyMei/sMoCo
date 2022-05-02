import json
import os
import math
import itertools
from tqdm import tqdm
import argparse

# 加载数据
def load(path, mode='list'):
    if mode == 'list':
        data = []
        with open(path, encoding='utf-8') as f:
            for line in tqdm(list(f)):
                data.append(json.loads(line))
    elif mode == 'dict':
        data = {}
        with open(path, encoding='utf-8') as f:
            for line in tqdm(list(f)):
                data.update(json.loads(line))
    return data

# 保存数据为文档
def dump(path, data, mode='json'):
    with open(path, 'w') as f:
        for line in tqdm(data):
            if mode == 'json':
                json.dump(line, f)
            elif mode == 'list':
                f.write(line)
            f.write('\n')
        f.close()

# 将子文件转换为json格式
def process_file(i, file='tmp/relations_pal/relation_pal'):
    path = file + str(i) + '.txt'
    save_path = file + str(i) + '.json'
    with open(save_path, 'w') as f_out:
        with open(path, encoding='utf-8') as f:
            for line in f:
                f_out.write(json.dumps(json.loads(line)))
                f_out.write('\n')
            f.close()

# 将字典点转换成列表
def splitDict(d):
    lists = []
    n = len(d) // len(d)  # length of smaller half
    i = iter(d.items())  # alternatively, i = d.iteritems() works in Python 2
    for x in range(len(d)):
        d = dict(itertools.islice(i, n))  # grab first n items
        lists.append(d)
    return lists

# 将数据文件分割成多个子文件方便并行处理
def splitfile(data, processor, path='tmp/id2w/id2w', mode='json'):
    l_data = len(data)
    size = math.ceil(l_data / processor)
    for i in range(processor):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < l_data else l_data

        # 保存文件
        if mode == 'json':
            file = path + str(i) + '.json'
            with open(file, 'w', encoding='utf-8') as f:
                for i in range(start, end):
                    f.write(json.dumps(data[i]))
                    f.write('\n')
                f.close()
        elif mode == 'txt':
            file = path + str(i) + '.txt'
            with open(file, 'w', encoding='utf-8') as f:
                for i in range(start, end):
                    f.write(data[i])
                f.close()
        print('{} Over!'.format(i))

    del data, l_data
    print('Split Over!')

# 去重
def del_duplicate(data):
    formatList = []
    for id in data:
        if id not in formatList:
            formatList.append(id)
    return formatList

def extractid(data):
    if data['text'][0] is '<':
        ent2id = data['text'].strip('<>')[3:]
    else:
        ent2id = data['text']
    return ent2id

def get_config():
    parser = argparse.ArgumentParser()

    # datasets
    parser.add_argument('--data_folder', default='datasets/webqsp/kb_01/', type=str)
    parser.add_argument('--train_data', default='train.json', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--template_folder', default='datasets/template_data/', type=str)
    parser.add_argument('--template_data', default='train_multi_hop.json', type=str)
    parser.add_argument('--save_path', default='datasets/new_samples/full/', type=str)
    parser.add_argument('--positive_num', default=3, type=int)

    # 超参数
    parser.add_argument('--eps', default=3, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    args = parser.parse_args()
    config = vars(args)

    print('-' * 10 + 'Experiment Config' + '-' * 10)
    for k, v in config.items():
        print(k + ': ', v)
    print('-' * 10 + 'Experiment Config' + '-' * 10 + '\n')

    return config

if __name__ == '__main__':
    # data = load('datasets/webqsp/kb_03/new_data.json')
    # negativate_data = load('datasets/webqsp/kb_01/new_negative_samples.json')
    # new_data = []
    # for line in data:
    #     if line not in negativate_data:print(question)
    #         new_data.append(line)
    # print(len(new_data))
    # dump('webqsp/kb_01/new_data.json', new_data)
    # from Truecaser import pre_truecaser
    # import spacy
    # cfg = get_config()
    # mode = 'train'
    # data_path = 'datasets/webqsp/full/{}.json'.format(mode)
    # data = load(data_path)
    # neg_questions = []
    # neg_samples = []
    # for line in data:
    #     entities = []
    #     question = pre_truecaser(line['question'])
    #     nlp = spacy.load('en_core_web_sm')
    #     doc = nlp(question)
    #     # print(len(doc.ents))
    #     # neg_question = []
    #     # neg_sample = []
    #     if len(doc.ents) == 0:
    #         print(question)
    #         neg_questions.append(question)
    #         neg_samples.append(line)
    #
    # dump('neg_questions.json', neg_questions)
    # dump('neg_samples.json', neg_samples)
    data = []
    with open('D:/论文代码/Knowledge-Aware-Reader/datasets/new_data/full/new_data3.json') as f:
        for line in tqdm(list(f)):
            data.append(json.loads(line))

    new_data = del_duplicate(data)
    print(len(new_data))
    dump('D:/论文代码/Knowledge-Aware-Reader/datasets/new_data/full/new_data4.json', new_data)
