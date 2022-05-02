import json
import networkx as nx
import pylab
import logging
import matplotlib
from tqdm import tqdm

# matplotlib.use('Agg')
with open('datasets/id2entity.json') as f:
    entityDic = json.load(f)
    f.close()

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


def searchId(keyword, Q_entities):
    keys = []
    for key, value in entityDic.items():
        if keyword.lower() == value.lower() and key in Q_entities:
            keys.append(key)
            return keys
    return ['empty']


def searchEnt(keyword, answer):
    if keyword in answer[0]:
        return answer[1]
    for key, value in entityDic.items():
        if keyword in key:
            return value
    return 'empty'


def two(list1, list2):
    res_lists = []
    for str1 in list1:
        res_list1 = []
        for str2 in list2:
            res_list2 = []
            if len(str1[0]) > 1 and str(type(str1[0])) != '<class \'dict\'>':
                res_list2.extend(str1[0])
            else:
                res_list2.append(str1)
            res_list2.append(str2)
            res_list1.append(res_list2)
        res_lists.append(res_list1)
    return res_lists


def cartesian_product(list_of_list):
    list1 = list_of_list[0]
    for tmp_list in list_of_list[1:]:
        list2 = tmp_list
        two_res_list = two(list1, list2)
        list1 = two_res_list
    return list1


def get_Qid2answer_path(data_path):
    data = []
    with open(data_path) as f:
        for line in tqdm(list(f)):
            data.append(json.loads(line))

    Qid2answers = []  # 存放路径
    questions = []

    for line in data:
        if len(line['entities']) == 0:
            print(line['question'])
        answers = []  # KG中所有答案的集合
        for answer in line['answers']:
            if answer['kb_id'][0] is '<':
                ent = answer['kb_id'].strip('<>')[3:]
                answers.append((ent, answer['text']))
            else:
                ent = answer['kb_id']
                answers.append((ent, ent))

        Q_ent = line['entities']

        edges = []  # 存放三元组中有关联的entity
        nodes = set()  # 三元组中所有节点

        all_triples = line['subgraph']['tuples']
        for tpl in all_triples:
            s, r, o = tpl
            if s['text'][0] is '<':
                s = s['text'].strip('<>')[3:]
            else:
                s = s['text']
            if o['text'][0] is '<':
                o = o['text'].strip("<>")[3:]
            else:
                o = o['text']
            edges.append([(s, o), tpl])  # 表示s,o之间存在边
            nodes.update({s, o})

        # 2.匹配ent_id，并确认该id存在于问题KG中
        # Qids = []
        # for ent in entities:
        #     Qid = searchId(ent, nodes)
        #     Qids.extend(Qid)

        # 3.将subgraph转换成图，采用networkx
        G = nx.Graph()
        G.add_nodes_from(nodes)  # 添加节点
        G.add_edges_from(edge[0] for edge in edges)  # 添加边

        # 4.匹配路径
        for Qid in Q_ent:
            Qid = Qid['text']
            if Qid[0] is '<':
                Qid = Qid.strip('<>')[3:]
            for answer in answers:
                if answer[0] in nodes:
                    try:
                        id_path = nx.dijkstra_path(G, source=Qid, target=answer[0])
                    except(nx.NetworkXNoPath, nx.NodeNotFound):
                        logging.warning("There is no path between two nodes: %s - %s " % (Qid, answer[0]))
                    Qid2answer = {}
                    Qid2answer['question'] = line['question']
                    Qid2answer['answer'] = answer[1]
                    Qid2answer.setdefault('path', [])
                    Qid2answer.setdefault('id2info', {})

                    target_paths = []
                    for i in range(len(id_path)-1):
                        path = (id_path[i], id_path[i+1])
                        tmp_path = []  # 存放同类型节点的路径
                        for edge in edges:
                            if path == edge[0]:
                                tmp_path.append(edge[1])
                        if len(tmp_path) != 0:
                            target_paths.append(tmp_path)

                    if len(target_paths) != 0:
                        Qid2answer['path'] = cartesian_product(target_paths)

                    for id in id_path:
                        Qid2answer['id2info'][id] = searchEnt(id, answer)

                else:
                    # print('{} Not in KG!'.format(answer))
                    continue
                if len(Qid2answer['path']) != 0:
                    Qid2answers.append(Qid2answer)

    return Qid2answers

def get_que_and_rel(data_path):

    data = get_Qid2answer_path(data_path)
    train_rels = {}
    for line in data:
        question = line['question']
        # print(question)
        q_node = list(line['id2info'])[0]
        answer = list(line['id2info'])[-1]
        relation = []
        if len(line['id2info']) > 2:
            for paths in line['path']:
                for path in paths:
                    if str(type(path)) == '<class \'dict\'>':
                        continue
                    for tpl in path:
                        s, r, o = tpl
                        rel = r['text']
                        relation.append(rel)
        else:
            for tpl in line['path']:
                s, r, o = tpl
                rel = r['text']
                relation.append(rel)

        question = question + ' [{}]'.format(q_node)
        if question not in list(train_rels):
            train_rels.setdefault(question, [])
        for rel in relation:
            if rel not in train_rels[question]:
                train_rels[question].append(rel)
    return train_rels

if __name__ == '__main__':
    from tqdm import tqdm
    train_rels = get_que_and_rel('C:/Users/Administrator/Desktop/model/new_data/full/new_data_1.json')
    with open('relation.json', 'w') as f_out:
        for k, v in train_rels.items():
            dict = {}
            dict[k] = v
            json.dump(dict, f_out)
            f_out.write('\n')
        f_out.close()

