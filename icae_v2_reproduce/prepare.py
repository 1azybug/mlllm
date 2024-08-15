import os
import random
from tqdm import tqdm
import json


with open('../compress/long_text.json', 'r', encoding='utf-8') as f:
    long_text_list =  json.load(f)


train_data = [{"text":text[:4096]} for text in long_text_list[1000:]]
eval_data = [{"text":text[:4096]} for text in long_text_list[:1000]]

with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False)
with open('eval_data.json', 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, ensure_ascii=False,indent=4)

with open('mini_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data[-256_000:], f, ensure_ascii=False)

with open('tiny_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data[-25600:], f, ensure_ascii=False)
