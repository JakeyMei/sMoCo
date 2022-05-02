import os
import json
import requests
from multiprocessing import Pool
from bs4 import BeautifulSoup
import multiprocessing as mp
from utils import splitDict, splitfile, load, dump

def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1',
        'Connection': 'close'
    }  # 模拟浏览器访问
    # requests.adapters.DEFAULT_RETRIES = 5
    response = requests.get(url, headers)
    html = response.text  # 获取网页源码
    return html

def process(path, pid):
    id2w = {}
    id2entity = {}
    # 加载数据
    id2w = load(path, mode='dict')

    # 处理数据：提取实体信息
    for key, value in id2w.items():
        value = value.strip('<>')
        ids = value[31:]

        url = 'https://www.wikidata.org/wiki/{}'.format(ids)
        soup = BeautifulSoup(get_html(url), 'html.parser')
        ent = soup.find('span', class_='wikibase-labelview-text')
        if ent:
            entity = ent.get_text()
        else:
            print('Not Found: {}'.format(ids))
            continue
        id2entity[key] = entity

    # 保存数据
    save_path = 'tmp/id2entity/id2entity' + str(pid) + '.json'
    dump(save_path, id2entity)

    print(str(pid) + ' processor over !')


def run(i):
    filename = 'tmp/id2w/id2w' + str(i) + '.json'
    process(filename, i)

if __name__ == "__main__":
    path = 'datasets/template_data/id2w.json'
    id2w = {}
    with open(path, encoding='utf-8') as f:
        id2w.update(json.load(f))
        f.close()

    processor = int(mp.cpu_count() * 0.7)
    data = splitDict(id2w)
    splitfile(data, processor)

    p = Pool(processor)
    for i in range(processor):
        p.apply_async(run, args=(i,))
        print(str(i) + ' processor started !')

    p.close()
    p.join()
    print("Process over!")

    # 整合
    id2entity_total = load('tmp/id2entity/id2entity3.json', mode='dict')
    for i in range(processor-1):
        path = 'tmp/id2entity/id2entity' + str(i) + '.json'
        file = os.path.join(os.getcwd(), path)
        id2entity = load(file, mode='dict')
        id2entity_total.update(id2entity)
    with open('datasets/template_data/id2entity.json', 'w') as f:
        json.dump(id2entity_total, f)
        f.close()