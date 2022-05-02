import json
import nltk
import numpy as np
import random
import torch

from collections import defaultdict
from tqdm import tqdm
from util import get_config
from util import load_dict
from util import load_documents
#from Qid2answer import get_que_and_rel

class DataLoader():
    def __init__(self, config, documents, mode='train'):
        self.mode = mode
        self.use_doc = config['use_doc']  # 默认为False
        self.use_inverse_relation = config['use_inverse_relation']
        self.max_query_word = config['max_query_word']  # 默认最大Q长度为10
        self.max_document_word = config['max_document_word']  # 默认document最大单词长度为50
        self.max_char = config['max_char']  # 默认字符最大长度为25
        self.documents = documents  # 知识库外文档
        # data_file为数据路径 = data_folder默认路径为datasets/webqsp/kb_03/ + 'mode_data'（train.json/test.json/dev.json）
        self.data_file = config['data_folder'] + config['{}_data'.format(mode)]
        self.batch_size = config['batch_size'] if mode == 'train' else config['batch_size']  # 批大小默认设置为8
        self.max_rel_words = config['max_rel_words']  # 默认所关联的单词最大为8
        self.type_rels = config['type_rels']  # 关系类别
        self.kb_ratio = config['kb_ratio']  # 默认为0，范围（0，1）

        # read all data
        # 加载数据，包含了document、subgraph、question、answer
        self.data = []
        with open(self.data_file) as f:
            for line in tqdm(list(f)):
                self.data.append(json.loads(line))
            f.close()

        self.new_data = []
        for batch_id in range(0, len(self.data), self.batch_size):  # 从0~2840间隔为8地提取batch_id
            batch = self.data[batch_id:batch_id + self.batch_size]  # 提取范围batch_id ~ (batch_id + 8)
            self.new_data.append(batch)

#        if self.use_inverse_relation:
#            self.questions2relation = get_que_and_rel(self.data_file)

        # word and kb vocab
        self.word2id = load_dict(config['data_folder'] + config['word2id'])  # KB单词word2id，默认为glove_vocab.txt
        self.relation2id = load_dict(config['data_folder'] + config['relation2id'])  # 关系集relation2id，默认为relations.txt
        self.id2relation = {i: relation for relation, i in self.relation2id.items()}  # id2relation
        self.entity2id = load_dict(config['data_folder'] + config['entity2id'])  # 实体集entity2id，默认为entities.txt
        self.id2entity = {i: entity for entity, i in self.entity2id.items()}  # id2entity

        self.rel_word_idx = np.load(config['data_folder'] + 'rel_word_idx.npy')

        # for batching
        self.max_local_entity = 0 # max num of candidates 初始化最大候选集
        self.max_relevant_docs = 0 # max num of retired documents 初始化最大关联文档
        self.max_kb_neighbors = config['max_num_neighbors'] # max num of neighbors for entity 实体最大关联数初始化为100
        self.max_kb_neighbors_ = config['max_num_neighbors'] # kb relations are directed
        self.max_linked_entities = 0 # max num of linked entities for each doc 每个文档的最大链接实体数初始化为0
        self.max_linked_documents = 50 # max num of linked documents for each entity 每个实体的最大链接文档数初始化为50
        self.max_local_relation = 0

        self.num_kb_relation = 2 * len(self.relation2id) if self.use_inverse_relation else len(self.relation2id)

        # get the batching parameters
        self.get_stats()

    # 在documents中提取有关联的实体
    def get_stats(self):
        if self.use_doc:
            # max_linked_entities
            self.useful_docs = {} # 过滤不带链接实体的文档
            for docid, doc in self.documents.items():
                linked_entities = 0
                if 'title' in doc:
                    linked_entities += len(doc['title']['entities'])
                    offset = len(nltk.word_tokenize(doc['title']['text']))
                else:
                    offset = 0
                for ent in doc['document']['entities']:
                    if ent['start'] + offset >= self.max_document_word:
                        continue
                    else:
                        linked_entities += 1
                if linked_entities > 1:
                    self.useful_docs[docid] = doc
                self.max_linked_entities = max(self.max_linked_entities, linked_entities)
            print('max num of linked entities: ', self.max_linked_entities)

        # decide how many neighbors should we consider
        # num_neighbors = []

        num_tuples = []
        
        # max_linked_documents, max_relevant_docs, max_local_entity
        # 从data中提取出所有的候选集和关联文档
        for line in tqdm(self.data):
            candidate_ents = set()  # 候选实体集
            rel_docs = 0  # 关联文档数

            # 候选entities集（candidate_ents） = 问题的entities + subgraph中的entities + document中的实体(use_doc)
            # question entity
            for ent in line['entities']:
                candidate_ents.add(ent['text'])  # 提取问题中的实体
            # kb entities
            for ent in line['subgraph']['entities']:
                candidate_ents.add(ent['text'])  # 提取subgraph中的实体

            num_tuples.append(line['subgraph']['tuples'])  # 提取中三元组

            if self.use_doc:
                # entities in doc
                # document中的实体
                for passage in line['passages']:
                    if passage['document_id'] not in self.useful_docs:
                        continue
                    rel_docs += 1
                    document = self.useful_docs[int(passage['document_id'])]
                    for ent in document['document']['entities']:
                        candidate_ents.add(ent['text'])
                    if 'title' in document:
                        for ent in document['title']['entities']:
                            candidate_ents.add(ent['text'])

            neighbors = defaultdict(list)  # 若key值不存在，则赋值为空列表[]
            neighbors_ = defaultdict(list)

            candidate_relations = set()
            for triple in line['subgraph']['tuples']:
                s, r, o = triple
                rel = r['text']
                candidate_relations.add(rel)
                # neighbors格式：s,(r,o)
                neighbors[s['text']].append((r['text'], o['text']))  # 将三元组中的text连接，去除实体编码
                # neighbors_格式：o,(r,s)
                neighbors_[o['text']].append((r['text'], s['text']))

            self.max_relevant_docs = max(self.max_relevant_docs, rel_docs)
            self.max_local_entity = max(self.max_local_entity, len(candidate_ents))  # 所提取实体的最大数
            self.max_local_relation = max(self.max_local_relation, len(candidate_relations))
            # self.max_kb_neighbors = max(self.max_kb_neighbors, len(neighbors))
            # self.max_kb_neighbors_ = max(self.max_kb_neighbors_, len(neighbors_))

        # np.save('num_neighbors_', num_neighbors)

        print('mean num of triples: ', len(num_tuples))

        print('max num of relevant docs: ', self.max_relevant_docs)
        print('max num of candidate entities: ', self.max_local_entity)
        print('max_num of neighbors: ', self.max_kb_neighbors)
        print('max_num of neighbors inverse: ', self.max_kb_neighbors_)
        print('max num of candidate relations:', self.max_local_relation)

    def batcher(self, shuffle=False):

        if shuffle:
            random.shuffle(self.new_data)  # 将data中的元素随机排序

        device = torch.device('cuda')

        # for batch_id in tqdm(range(0, len(self.data), self.batch_size)):  # 从0~2840间隔为8地提取batch_id
        #     batch = self.data[batch_id:batch_id + self.batch_size]  # 提取范围batch_id ~ (batch_id + 8)
        for batch in tqdm(self.new_data):

            batch_size = len(batch)  # 每8段passage用于训练
            # np.full/np.zeros表示创建数组
            questions = np.full((batch_size, self.max_query_word), 1, dtype=np.longlong)
            documents = np.full((batch_size, self.max_relevant_docs, self.max_document_word), 1, dtype=np.longlong)
            entity_link_documents = np.zeros((batch_size, self.max_local_entity, self.max_linked_documents, self.max_document_word), dtype=np.longlong)
            entity_link_doc_norm = np.zeros((batch_size, self.max_local_entity, self.max_linked_documents, self.max_document_word), dtype=np.longlong)
            documents_ans_span = np.zeros((batch_size, self.max_relevant_docs, 2), dtype=int)
            # 由-1填充的关系集数组
            entity_link_ents = np.full((batch_size, self.max_local_entity, self.max_kb_neighbors_), -1, dtype=np.longlong)  # incoming edges
            # 由0填充的关系集数组
            entity_link_rels = np.zeros((batch_size, self.max_local_entity, self.max_kb_neighbors_), dtype=np.longlong)
            candidate_entities = np.full((batch_size, self.max_local_entity), len(self.entity2id), dtype=np.longlong)  # batch_size * candidate_ents
            ent_degrees = np.zeros((batch_size, self.max_local_entity), dtype=np.longlong)
            true_answers = np.zeros((batch_size, self.max_local_entity), dtype=float)  # batch_size * candidate_ents
            query_entities = np.zeros((batch_size, self.max_local_entity), dtype=float)  # batch_size * candidate_ents
            que2rel = np.zeros((batch_size, self.max_local_relation), dtype=float)  # 问题对应的关系
            # que2rel = np.zeros((batch_size, len(self.relation2id)), dtype=float)
            answers_ = []
            questions_ = []
            # negative_samples = []  # 无法找到true_answer特征的样本数据

            # 取样策略：提取answer集合和question集合
            for i, sample in enumerate(batch):  # i:passage_id，sample:passage

                doc_global2local = {}
                # answer set
                answers = set()
                for answer in sample['answers']:
                    keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'  # 如果answer中的kb_id是int，则keyword赋值为文本
                    answers.add(self.entity2id[answer[keyword]])  # 从entity2id中提取出answer中的实体，保存为answer集合

                if self.mode != 'train':
                    answers_.append(list(answers))
                    questions_.append(sample['question'])
                
                # candidate entities, linked_documents
                # 候选text集（candidates） = 问题对应的实体 + subgraph中的text对应的实体 (+ 文档实体)
                candidates = set()
                query_entity = set()
                ent2linked_docId = defaultdict(list)
                q_ent = ' '
                for ent in sample['entities']:
                    q_ent = ent['text'].strip("<>")[3:]
                    candidates.add(self.entity2id[ent['text']])
                    query_entity.add(self.entity2id[ent['text']])  # 问题实体
                for ent in sample['subgraph']['entities']:
                    candidates.add(self.entity2id[ent['text']])

                if self.use_doc:
                    for local_id, passage in enumerate(sample['passages']):
                        if passage['document_id'] not in self.useful_docs:
                            continue
                        doc_id = int(passage['document_id'])
                        doc_global2local[doc_id] = local_id
                        document = self.useful_docs[doc_id]
                        for word_pos, word in enumerate(['<bos>'] + document['tokens']):
                            if word_pos < self.max_document_word:
                                documents[i, local_id, word_pos] = self.word2id.get(word, self.word2id['<unk>'])
                        for ent in document['document']['entities']:
                            if self.entity2id[ent['text']] in answers:
                                documents_ans_span[i, local_id, 0] = min(ent['start'] + 1, self.max_document_word-1)
                                documents_ans_span[i, local_id, 1] = min(ent['end'] + 1, self.max_document_word-1)
                            s, e = ent['start'] + 1, ent['end'] + 1
                            ent2linked_docId[self.entity2id[ent['text']]].append((doc_id, s, e))
                            candidates.add(self.entity2id[ent['text']])
                        if 'title' in document:
                            for ent in document['title']['entities']:
                                candidates.add(self.entity2id(ent['text']))

                # kb information
                connections = defaultdict(list)

                if self.kb_ratio and self.mode == 'train':
                    all_triples = sample['subgraph']['tuples']
                    random.shuffle(all_triples)  # 将三元组顺序打乱
                    num_triples = len(all_triples)
                    all_triples = all_triples[:int(num_triples * self.kb_ratio)]  # 随机抽取剩余的kb

                else:
                    all_triples = sample['subgraph']['tuples']

                # 问题对应关系（标签）

                candidate_relations = set()

                for tpl in all_triples:
                    s, r, o = tpl

                    # only consider one direction of information propagation
                    # connections格式：{o: [r,s]}
                    connections[self.entity2id[o['text']]].append((self.relation2id[r['text']], self.entity2id[s['text']]))

                    # 若关系在type_rels中，则connections格式为：{s: [r,o]}
                    if r['text'] in self.type_rels:
                        connections[self.entity2id[s['text']]].append((self.relation2id[r['text']], self.entity2id[o['text']]))

                    relation = r['text']
                    candidate_relations.add(relation)

                # question to relation
#                if self.use_inverse_relation:
#                    que = sample['question'] + ' [{}]'.format(q_ent)
                    # if que in self.questions2relation.keys():
                    #     rel_ids = [self.relation2id[rel] for rel in self.questions2relation[que]]
                    #     rel_onehot = self.toOneHot(rel_ids)
                    #     que2rel[i, :] = rel_onehot
#                    for j, rel in enumerate(candidate_relations):
#                        if que in self.questions2relation.keys():
#                            if rel in self.questions2relation[que]:
#                                que2rel[i, j] = 1.0


                # used for updating entity representations
                # 更新实体表示
                ent_global2local = {}
                candidates = list(candidates)

                # if len(candidates) == 0:
                    # print('No entities????')
                    # print(sample)

                for j, entid in enumerate(candidates):  # j表示id(行号)，entid表示text id
                    if entid in query_entity:  # 若与问题相匹配
                        query_entities[i, j] = 1.0  # 令query_entities数组中该值为1，否则为0
                    candidate_entities[i, j] = entid
                    ent_global2local[entid] = j  # 使用字典存放实体，key值为其对应的id号
                    if entid in answers:  # 若该实体与答案匹配，则令true_answers数组中该值为1，否则为0
                        true_answers[i, j] = 1.0

                    # document中寻找相关实体
                    for linked_doc in ent2linked_docId[entid]:
                        start, end = linked_doc[1], linked_doc[2]
                        if end - start > 0:
                            entity_link_documents[i, j, doc_global2local[linked_doc[0]], start:end] = 1.0
                            entity_link_doc_norm[i, j, doc_global2local[linked_doc[0]], start:end] = 1.0
                # if flag == 0:
                    # del sample
                    # negative_samples.append(sample)

                # kb库中找寻相关实体
                for j, entid in enumerate(candidates):
                    for count, neighbor in enumerate(connections[entid]):
                        if count < self.max_kb_neighbors_:
                            r_id, s_id = neighbor
                            # convert the global ent id to subgraph id, for graph convolution
                            s_id_local = ent_global2local[s_id]
                            entity_link_rels[i, j, count] = r_id
                            entity_link_ents[i, j, count] = s_id_local
                            ent_degrees[i, s_id_local] += 1

                # questions
                # 问题单词匹配
                for j, word in enumerate(sample['question'].split()):
                    if j < self.max_query_word:
                        if word in self.word2id:
                            questions[i, j] = self.word2id[word]
                        else: 
                            questions[i, j] = self.word2id['<unk>']

            if self.use_doc:
                # exact match features for docs
                d_cat = documents.reshape((batch_size, -1))
                em_d = np.array([np.isin(d_, q_) for d_, q_ in zip(d_cat, questions)], dtype=int)  # exact match features
                em_d = em_d.reshape((batch_size, self.max_relevant_docs, -1))

            batch_dict = {
                'questions': questions, # (B, q_len)
                'candidate_entities': candidate_entities,
                'entity_link_ents': entity_link_ents,  # 与候选实体存在关系的实体id
                'answers': true_answers,
                'query_entities': query_entities,  # 问题实体
                'answers_': answers_,
                'questions_': questions_,
                'rel_word_ids': self.rel_word_idx, # (num_rel+1, word_lens)
                'entity_link_rels': entity_link_rels, # (bsize, max_num_candidates, max_num_neighbors)  # 存放关系
                'ent_degrees': ent_degrees,
            }

            if self.use_doc:
                batch_dict['documents'] = documents
                batch_dict['documents_em'] = em_d
                batch_dict['ent_link_doc_spans'] = entity_link_documents
                batch_dict['documents_ans_span'] = documents_ans_span
                batch_dict['ent_link_doc_norm_spans'] = entity_link_doc_norm

#            if self.use_inverse_relation:
#                batch_dict['que2rel'] = que2rel  # (B, max_local_candidates)

            for k, v in batch_dict.items():
                if k.endswith('_'):
                    batch_dict[k] = v
                    continue
                if not self.use_doc and 'doc' in k:
                    continue
                batch_dict[k] = torch.from_numpy(v).to(device)

            yield batch_dict


if __name__ == '__main__':
    cfg = get_config()
    documents = load_documents(cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
    # cfg['batch_size'] = 2
    train_data = DataLoader(cfg, documents)
    # build_squad_like_data(cfg['data_folder'] + cfg['{}_data'.format(cfg['mode'])], cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
