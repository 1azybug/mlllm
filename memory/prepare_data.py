from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json


dataset = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
print(dataset)
for i,example in enumerate(dataset):
    print(f"example {i}:")
    print(example)
    break

import json
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("deepmind/narrativeqa", split="train", streaming=True)

# 创建一个空列表来存储样本
samples = []

# 遍历数据集
for i, example in enumerate(dataset):
    # 将样本添加到列表中
    samples.append(example)
    
    # 为了示例，这里只保存前几个样本
    if i >= 4:  # 保存前5个样本
        break

# 指定要写入的JSON文件路径
output_file = "narrativeqa_samples.json"

# 将样本列表写入到JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)

print(f"Samples have been saved to {output_file}")

    



# import json
# from datasets import load_dataset

# # 加载数据集
# dataset = load_dataset("hotpotqa/hotpot_qa", 'fullwiki',trust_remote_code=True)

# # 创建一个空列表来存储样本
# samples = []

# # 遍历数据集
# for i, example in enumerate(dataset):
#     # 将样本添加到列表中
#     samples.append(example)
    
#     # 为了示例，这里只保存前几个样本
#     if i >= 4:  # 保存前5个样本
#         break

# # 指定要写入的JSON文件路径
# output_file = "hotpot_qa_distractor.json"

# # 将样本列表写入到JSON文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(samples, f, ensure_ascii=False, indent=4)

# print(f"Samples have been saved to {output_file}")