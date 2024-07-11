from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='Directory including the configuration file')
    return parser.parse_args()


def get_long_text_list(dataset):
    # cache long text for preventing full dataset traversal on each preparation. 
    if os.path.exists('long_text.json'):
        with open('long_text.json', 'r', encoding='utf-8') as f:
            long_text_list =  json.load(f)
        return long_text_list

    long_text_list = []
    for example in tqdm(dataset, desc="Processing examples"):
        if len(example["text"])>=512*8:
            long_text_list.append(example["text"])
        
    with open('long_text.json', 'w', encoding='utf-8') as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    return long_text_list

    


def get_examples(model_id, dataset_repo="DKYoon/SlimPajama-6B",hf_token=None, token_num=1_000_000_000,min_len=512):
    
    model_name = model_id.split('/')[-1]
    train_data_name = "train_"+model_name+"_"+str(token_num)+f"token_len-{min_len}.pt"
    eval_data_name = "eval_"+model_name+"_"+str(token_num)+f"token_len-{min_len}.pt"
    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

#     model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="cpu",
#     token=hf_token
# )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    dataset = load_dataset(dataset_repo, split="train", streaming=True)
    
    
    examples = []

    long_text_list = get_long_text_list(dataset)

    for text in tqdm(long_text_list, desc="Processing examples"):
        
        ids = tokenizer(text)["input_ids"]
        
        if len(ids)<min_len*2:
            continue
        
        # half for prefix, half for LM
        last_start = len(ids)-min_len*2
        random_start = random.randint(0, last_start)
        
        inputs = torch.LongTensor(ids[random_start:random_start+min_len])
        lm_target = torch.LongTensor(ids[random_start+min_len:random_start+2*min_len])
        examples.append({"inputs":inputs,"lm_target":lm_target})
        
        if len(examples)*min_len>=token_num:
            break
        
    

    # 1k for validation
    torch.save(examples[1000:], train_data_name)
    torch.save(examples[:1000], eval_data_name)
    
    return examples[1000:], examples[:1000]
    
if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)
    
    training_config = config["training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]
    
    print(config["data_config"])
    train_examples, eval_examples = get_examples(**config["data_config"])
    print(len(train_examples))
    print(train_examples[50])

"""
python prepare_data.py --work_dir CompressLLM

unset HF_HUB_OFFLINE
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 python prepare_data.py --work_dir compressLLM_len-510_ratio-15
HF_ENDPOINT=https://hf-mirror.com python prepare_data.py --work_dir compressLLM_len-510_ratio-15
"""