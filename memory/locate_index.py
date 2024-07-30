from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json
import numpy as np


class Locator:

    def __init__(self, model_id, device_rank, segment_len):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{device_rank}",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.segment_len = segment_len
        self.eos_id = self.tokenizer.eos_token_id
        self.model.eval()
        print(f"eos_id:{self.eos_id}")

    def locate_index(example):
        context = example["document"]["text"]
        question = example["question"]["text"]
        answers = example["answers"]
        answers = [item['text'] for item in answers]

        context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        question_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        answers_ids = [self.tokenizer(answer, add_special_tokens=False)["input_ids"] for answer in answers]

        context_prefix = self.tokenizer('Snippet: ', add_special_tokens=False)["input_ids"]
        question_prefix = self.tokenizer('\nQuestion: ', add_special_tokens=False)["input_ids"]
        answer_prefix = self.tokenizer('\nAnswer: ', add_special_tokens=False)["input_ids"]

        print(len(context_ids))
        [context_ids[i:i + segment_len] for i in range(0, len(context_ids), self.segment_len)]

        example["chunks"] = []

        with torch.no_grad():
            for i in range(0, len(context_ids), self.segment_len):
    
                print(f"Page {i}:")
                chunk = context_ids[i:i + segment_len]
                chunk_2 = context_ids[i:i + segment_len*2]
                input_ids = context_prefix + chunk_2 + question_prefix + question_ids + answer_prefix
                # input_lables = [-100 for x in input_ids]

                past_key_values = None
                outputs = self.model(
                input_ids=torch.LongTensor(input_ids[:-1]),
                past_key_values=past_key_values,
                use_cache=True
            )
                past_key_values = outputs.past_key_values
                sum_loss = 0.0
                sum_tokens = 0
                for answer_ids in answers_ids:
                    print(f"past_key_values:{past_key_values}")
                    target_ids = answer_ids + [self.eos_id]  # 2 is <eos>
                    target_label = answer_ids + [self.eos_id]
                    all_ids = input_ids[-1:] + target_ids[:-1]
                    all_label = target_label
                
                    outputs = self.model(
                    input_ids=torch.LongTensor(all_ids),
                    labels = torch.LongTensor(all_label),
                    past_key_values=past_key_values,
                    use_cache=True
                )

                    sum_loss += outputs.loss.item()*len(all_label)
                    sum_tokens = len(all_label)
                avg_loss = sum_loss/sum_tokens
                ppl = np.exp(avg_loss)
                print(f"ppl:{ppl}")
                example["chunks"].append({
                    "page":i,
                    "chunk":self.tokenizer.decode(chunk_2, skip_special_tokens=True),
                    "ppl":x
                    })

        example["sorted_chunks"] = sorted(example["chunks"], key=lambda x: x["ppl"])
        return example

            
                    

if __name__ == "__main__":
    with open('./narrativeqa_samples.json', 'r', encoding='utf-8') as f:
        examples =  json.load(f)

    for i in range(len(examples)):
        examples[i]=locate_index(examples[i])


    with open('./narrativeqa_samples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)  
