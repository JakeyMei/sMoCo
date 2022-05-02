# 环境和数据集
- 环境
  ```python
  pip install -r requirements.txt
  ```
- 数据集

  ```shell
  cd datasets && wget https://sites.cs.ucsb.edu/~xwhan/datasets/webqsp.tar.gz && tar -xzvf webqsp.tar.gz && cd ..
  ```

# 运行



- 找出实体ID对应的实体描述 **id2entity.py**

    - [Freebase/Wikidata Mappings](https://developers.google.com/freebase)

    

- 提取rel_id对应的关系描述 **relation.py**
  
    - [freepal-dataset.json.bz2](https://free-pal.appspot.com/)

  
  
- 问题模板生成 **question_template.py  questions_generation.py**

    

- 问题，答案路径生成 **Qid2answer.py 或者 Qid2answer2.py**

    - 命名实体识别模型权重[**English_distributions.obj.zip**](https://github.com/nreimers/truecaser/releases)

    

- 单跳数据生成 **one_hop.py** 

    

- 多跳数据生成 **multi_hop.py** 

    

- **问题生成**：数据排序**rank_data.py** , 调整原始数据集顺序, 新训练数据（[1-4：原始训练数据]， [5-batch_size：新生成数据]）

    ```python
	    python rank_data.py --data_folder datasets/webqsp/kb_01/ --save_path datasets/new_samples/kb_01/
    ```

