import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

hop = 3
file_name = 'full'  # full kb05
dataset = 'datasets/MetaQA/{}/{}-hop/'.format(file_name, hop)
# dataset = 'datasets/webqsp/'
rel_path = dataset + '/relations.txt'
save_path = dataset

word_counter = []

# data_emb = dataset + '/entities.txt'
data_emb = dataset + '/vocab.txt'
# load original vocab
with open(data_emb, encoding='utf-8') as f:
    for line in f.readlines():
        word_counter.append(line.strip())

rel_words = []
max_num_words = 0
all_relations = []

# how to split the relation
# 提取关系编码
if 'webqsp' in dataset:
    with open(rel_path) as f:
        first_line = True
        for line in tqdm(f.readlines()):
            if first_line:
                first_line = False
                continue
            line = line.strip()
            all_relations.append(line)
            line = line[1:-1]
            fields = line.split('.')
            words = fields[-2].split('_') + fields[-1].split('_')
            max_num_words = max(len(words), max_num_words)
            rel_words.append(words)
            word_counter += words
elif 'MetaQA' in dataset:
    with open(rel_path) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            all_relations.append(line)
            words = line.split('_')
            max_num_words = max(len(words), max_num_words)
            rel_words.append(words)
            word_counter += words

print('max_num_words: ', max_num_words)

word_counter = nlp.data.count_tokens(word_counter)
if 'vocab' in data_emb:
    print('glove 300')
    glove_emb = nlp.embedding.create('glove', source='glove.6B.300d')
else: 
    print('glove 100')
    glove_emb = nlp.embedding.create('glove', source='glove.6B.100d')
vocab = nlp.Vocab(word_counter)
vocab.set_embedding(glove_emb)

emb_mat = vocab.embedding.idx_to_vec.asnumpy()
if 'vocab' in data_emb:
    np.save(save_path + '/glove_word_emb', emb_mat)
    with open(save_path + '/glove_vocab.txt', 'w', encoding='utf-8') as g:
        g.write('\n'.join(vocab.idx_to_token))
else:
    np.save(save_path + '/entity_emb_100d', emb_mat)    

# assert False

# rel_word_ids = np.ones((len(rel_words) + 1, max_num_words), dtype=np.longlong) # leave the first 1 for padding relation
# rel_emb_mat = []
# for rel_idx, words in enumerate(rel_words):
#     for i, word in enumerate(words):
#         rel_word_ids[rel_idx + 1, i] = vocab.token_to_idx[word]

# np.save(save_path + '/rel_word_idx', rel_word_ids)

# all_relations = ['pad_rel'] + all_relations
# with open(rel_path, 'w') as g:
#     g.write('\n'.join(all_relations))



