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


def get_examples_list(instruction_dataset_repo, split):
    # cache long text for preventing full dataset traversal on each preparation. 
    if os.path.exists(f'{split}_instruction_dataset.json'):
        with open(f'{split}_instruction_dataset.json', 'r', encoding='utf-8') as f:
            examples_list =  json.load(f)
        return examples_list

    dataset = load_dataset(instruction_dataset_repo, split=split, streaming=True)

    examples_list = []
    for example in tqdm(dataset, desc="Processing examples"):
        
        examples_list.append(example)
        
    with open(f'{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list

    
def get_ids(examples_list, tokenizer, min_len, split):

    examples = []
    minn = 9999
    maxn = 0
    for example in tqdm(examples_list, desc="Processing examples"):
        
        context_ids = tokenizer(example["input"], add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example["answer"], add_special_tokens=False)["input_ids"]
        
        all_prompt_ids = tokenizer("### Context:\n")["input_ids"] + context_ids \
                    + tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt_ids \
                    + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]

        all_response_ids = answer_ids + tokenizer("</s>", add_special_tokens=False)["input_ids"]

        # print("inputs:|", all_prompt_ids)
        # print("outputs:|", all_response_ids)
        # decoded_inputs = tokenizer.decode(all_prompt_ids, skip_special_tokens=False)
        # decoded_outputs = tokenizer.decode(all_response_ids, skip_special_tokens=False)
        # print("inputs:|", decoded_inputs)
        # print("outputs:|", decoded_outputs)

        instruction_target = [-100 for x in all_prompt_ids] + [x for x in all_response_ids]
        instruction_target = instruction_target[1:]

        if split == 'train':
            all_ids = all_prompt_ids + all_response_ids
        else:
            all_ids = all_prompt_ids

        minn = min(minn, len(all_ids))
        maxn = max(maxn, len(all_ids))

        inputs = torch.LongTensor(all_ids[:min_len])

        if len(all_ids) >= min_len+2:  # will drop the end token and leave one token
            lm_target = torch.LongTensor(all_ids[min_len:])
        else:
            lm_target = None
        
        instruction_target = torch.LongTensor(instruction_target)

        if split == "test":
            examples.append({"input_ids":inputs,"lm_targets":lm_target})
        else:
            examples.append({"input_ids":inputs,"lm_targets":lm_target,
                            "instruction_target":instruction_target})
    print(f"len range: [{minn}:{maxn}]")
    return examples

def get_examples(model_id, instruction_dataset_repo="sggetao/PwC",hf_token=None, token_num=1_000_000_000,min_len=512, dataset_repo=None):
    
    model_name = model_id.split('/')[-1]
    train_data_name = "train_"+model_name+f"_len{min_len}_instruction.pt"
    eval_data_name = "eval_"+model_name+f"_len{min_len}_instruction.pt"
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
    
    train_examples_list = get_examples_list(instruction_dataset_repo, split="train")
    test_examples_list = get_examples_list(instruction_dataset_repo, split="test")


    train_data = get_ids(train_examples_list, tokenizer, min_len, split="train")
    test_data = get_ids(test_examples_list, tokenizer, min_len, split="test")
    
    torch.save(train_data, train_data_name)
    torch.save(test_data, eval_data_name)
    
    return train_data, test_data
    
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
    print(len(eval_examples))
    print(eval_examples[50])

"""

unset HF_HUB_OFFLINE
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 python vanilla_prepare_data.py --work_dir debug_CompressLLM_wo-cmp
HF_ENDPOINT=https://hf-mirror.com python vanilla_prepare_data.py --work_dir debug_CompressLLM_wo-cmp
"""