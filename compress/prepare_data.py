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

def get_examples(model_id, dataset_repo="DKYoon/SlimPajama-6B",hf_token=None, token_num=1_000_000_000,min_len=512):
    
    model_name = model_id.split('/')[-1]
    train_data_name = "train_"+model_name+"_"+str(token_num)+f"token_len-{min_len}.pt"
    eval_data_name = "eval_"+model_name+"_"+str(token_num)+f"token_len-{min_len}.pt"
    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    token=hf_token
)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    dataset = load_dataset(dataset_repo, split="train", streaming=True)
    
    
    examples = []
    for example in tqdm(dataset, desc="Processing examples"):
        if len(example["text"])<min_len*8:
            continue
        
        ids = tokenizer(example["text"])["input_ids"]
        
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
    
    train_examples, eval_examples = get_examples(**config["data_config"])
    print(len(train_examples))
    print(train_examples[50])

"""
python prepare_data.py --work_dir CompressLLM
"""