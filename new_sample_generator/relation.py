import json
import os
import bz2
import re
import multiprocessing as mp
from multiprocessing import Pool
from utils import load, dump, process_file, splitfile

# 保存字典数据
def dump_dict(data, save_path):
    with open(save_path, 'w') as f:
        for key, value in data.items():
            dict = {}
            dict[key] = value
            f.write(json.dumps(dict))
            f.write('\n')
        f.close()
    print('Save Done')

# 提取rel_id对应的关系描述
def process_rel(path, pid):
    relation = {}  # 存放匹配成功的关系和其特征
    relations_pal_ = load(path)

    for line in relations_pal_:
        feature = line['feature']
        patt = re.compile(r'\[\d.*?\]', re.S)
        _ = patt.findall(feature)
        feature = feature.replace(_[0], '')[:-1]  # 提取特征描述
        rel_ = line['relationCount']

        for rel in rel_:
            relation_pal = rel['relation'][3:].split('..')
            for key in relation_pal:
                if key in relations_:
                    relation.setdefault(key, [])
                    relation[key].append(feature)

    save_path = 'tmp/relations/relation' + str(pid) + '.json'
    dump_dict(relation, save_path)
    del relation
    print("Process_{} over!".format(str(pid)))

def run(i):
    print('Start....')
    path = 'tmp/relations_pal/relations_pal' + str(i) + '.json'
    process_rel(path, i)

if __name__ == "__main__":
    # 获取关系
    # with open('datasets/webqsp/full/relations.txt', encoding='utf-8') as f:
    #     for line in list(f)[1:]:
    #         line = line[4:-2]
    #         print(line)
    #         relations_.append(line)
    # dump('tmp/relations.json', relations_)


    relations_pal = []  # freepal关系集
    dataset_dir = 'tmp/relations.json'
    relations_ = load(dataset_dir)

    freepal_dir = 'tmp/freepal-dataset.json.bz2'
    with bz2.open(freepal_dir, 'rb') as f:
        for line in list(f):
            line = line.decode()
            relations_pal.append(line)
        f.close()
    print('freepal load Done!')

    processor = int(mp.cpu_count() * 0.7)
    # 将文件分为子文件方便并行处理
    splitfile(relations_pal, processor, path='tmp/relations_pal/relation_pal', mode='txt')

    p = Pool(processor)
    for i in range(processor):
        process_file(i, file='tmp/relations_pal/relation_pal')  # 文件格式转换
        print(str(i) + ' processor started !')

    for i in range(processor):
        p.apply_async(run, args=(i,))  # 提取rel_id对应的关系描述
        print(str(i) + ' processor started !')

    p.close()
    p.join()
    print("Process over!")

    # 整合
    relations_path = 'tmp/relations/relation3.json'
    relations = load(relations_path, mode='dict')

    for i in range(processor - 1):
        filename = 'tmp/relations/relation' + str(i) + '.json'
        file = os.path.join(os.getcwd(), filename)
        relation = load(file, mode='dict')
        for key_total, value_total in relations.items():
            for key, value in relation.items():
                if key == key_total:
                    value_total.extend(value)

    dump_dict(relations, save_path='datasets/template_data/relations.json')

