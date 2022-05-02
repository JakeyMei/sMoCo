### Requirements
* ``PyTorch 1.0.1``
* ``tensorboardX``
* ``tqdm``
* ``gluonnlp``
* ``requirements.txt``

### WebQSP

#### Prepare data

```
mkdir datasets && cd datasets && wget https://sites.cs.ucsb.edu/~xwhan/datasets/webqsp.tar.gz && tar -xzvf webqsp.tar.gz && cd ..
```
- [WebQSP-expand_data & train-model](https://github.com/JakeyMei/test/releases/tag/v1.0)

- you can also follow the following commands to expand new samples：
```
cd new_sample_generator
python rank_data.py --data_folder ../datasets/webqsp/kb_01/ --save_path ../datasets/new_samples/kb_01/
```



#### Full KB setting
```
CUDA_VISIBLE_DEVICES=0 python train.py --max_num_neighbors 50 --label_smooth 0.1 --data_folder datasets/webqsp/full/ --train_data new_data.json --model_id full --weight 0.2 --batch_size 16 --use_smoco --pos_mode hard
```

#### Incomplete KB setting
Note: The Hits@1 should match or be slightly better than the number reported in the paper. More tuning on threshold should give you better F1 score. 
##### 30% KB

```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id kb_03 --max_num_neighbors 50 --use_doc --data_folder datasets/webqsp/kb_03/ --eps 0.05 --train_data new_data.json --weight 0.2 --batch_size 16 --use_smoco --pos_mode hard
```

##### 10% KB
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id kb_01 --max_num_neighbors 50 --use_doc --data_folder datasets/webqsp/kb_01/ --eps 0.05 --train_data new_data.json --weight 0.2 --batch_size 16 --use_smoco --pos_mode hard
```
##### 50% KB

```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id kb_05 --num_layer 1 --max_num_neighbors 100 --use_doc --data_folder datasets/webqsp/kb_05/ --eps 0.05 --seed 3 --hidden_drop 0.05 --train_data new_data.json --weight 0.2 --batch_size 16 --use_smoco  --pos_mode hard
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=0 python train.py --max_num_neighbors 50 --label_smooth 0.1 --data_folder datasets/webqsp/full/ --train_data new_data.json --model_id full --weight 0.2 --batch_size 16 --use_smoco --pos_mode hard --mode test
```

