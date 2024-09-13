# vanilla_*.py modified from instruction_*.py

import json
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
from nltk.translate.bleu_score import sentence_bleu

from torch.nn import DataParallel
import torch.multiprocessing as mp

from vanilla_prepare_data import get_examples
from vanilla_modeling import get_model, save_adapter, load_adapter
from vanilla_dataloader import get_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='Directory including the configuration file')
    parser.add_argument('--batch_size', type=int, required=True, help='total batch size')
    return parser.parse_args()

class Evaluator:

    def __init__(self, config, work_dir, batch_size):
        self.config = config
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.device_count = torch.cuda.device_count()

    def draw_loss(self):
        with open(os.path.join(self.work_dir,"instruction_info.json")) as f:
            info_list=json.load(f)

        lm_loss_values = [entry['training_loss']["lm_loss"] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]
        
        plt.figure(figsize=(10, 5))
        plt.plot(step_values, lm_loss_values, label="lm_loss")

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(self.work_dir)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.work_dir, 'instruction_training_loss.png'))
        plt.close()


    def draw_ema_loss(self, alpha=0.9):

        def exponential_moving_average(values,alpha):
            ema = [values[0]]  # 初始化EMA的第一个值为原始值的第一个值
            for value in values[1:]:
                ema.append(alpha * value + (1 - alpha) * ema[-1])
            return ema

        with open(os.path.join(self.work_dir,"instruction_info.json")) as f:
            info_list=json.load(f)

        lm_loss_values = [entry['training_loss']["lm_loss"] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]

        
        plt.figure(figsize=(10, 5))
        plt.plot(step_values, exponential_moving_average(lm_loss_values,alpha=alpha), label="lm_loss")

        plt.xlabel("step")
        plt.ylabel(f"loss(ema_alpha={alpha})")
        plt.title(f"{self.work_dir}_loss(ema_alpha={alpha})")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.work_dir, 'instruction_ema_training_loss.png'))
        plt.close()

    def evaluate(self, rank=0):
        # rank:gpu_device_id
        training_config = self.config["training_config"]
        task_config = self.config["task_config"]
        train_examples, eval_examples = get_examples(**self.config["data_config"])

        example_num_per_gpu = len(eval_examples)//torch.cuda.device_count()
        # assert example_num_per_gpu*torch.cuda.device_count() == len(eval_examples)

        if rank <= self.device_count-2:
            eval_examples = eval_examples[rank*example_num_per_gpu:(rank+1)*example_num_per_gpu]
        else: # last gpu deal the left data.
            eval_examples = eval_examples[rank*example_num_per_gpu:]

        print(f"[INFO] GPU{rank}: eval_examples[{rank*example_num_per_gpu}:{rank*example_num_per_gpu+len(eval_examples)}], nums:{len(eval_examples)}")

        dataset = get_dataset(task_config["task_type"], eval_examples, batch_size=self.batch_size)
        loader = DataLoader(dataset, batch_size=None)
        
        model = get_model(training_config["model_id"], task_config, rank)
        model = load_adapter(model, save_path_and_name=self.work_dir+'/instruction_adapter.pt', log=False)
        model.eval()

        info_list=[]
        with torch.no_grad():
            for inputs in tqdm(loader,total=len(eval_examples)//self.batch_size):
                # inputs = {key:value.to(rank) for key,value in inputs.items()}
                inputs = {key:value.to(rank) if value is not None else None for key,value in inputs.items()}
                # output = model(inputs=inputs)
                generate_text = model.lm_inference(inputs,segment_size=task_config["segment_size"])
                # print(inputs['ae_targets'].size())
                # print("teacher forcing generate:", torch.argmax(output["logits"], dim=-1).tolist()) # B,S,V -> B,S
                # print("Auto-regressive generate:", generate_text)
                # print("target:", inputs['ae_targets'].tolist())
                # generate_text = tokenizer.decode(generate_text, skip_special_tokens=False)
                # bleu4 = sentence_bleu([inputs['ae_targets'].tolist()], generate_text, weights=(0.25, 0.25, 0.25, 0.25))
                # print(f"BLEU-4:",bleu4*100)
                # print('gen:', generate_text)
                
                info_list.append({"generate_text":generate_text})

        with open(self.work_dir+f'/instruction_eval_info_list_{rank}.json', 'w', encoding='utf-8') as f:
            json.dump(info_list, f, ensure_ascii=False)
        
        
        
                
    def run(self, rank):
        # draw training loss
        if rank==0:
            self.draw_loss()
            self.draw_ema_loss(alpha=0.1)
        self.evaluate(rank)


def evaluate(rank, args, world_size):

    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.work_dir+f'/instruction_evaluate_info_rank{rank}.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    evaluator = Evaluator(config, args.work_dir, args.batch_size)
    evaluator.run(rank)


# Launch multi-process eval
if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()

    if not os.path.exists(args.work_dir+f'/instruction_eval_info_list_0.json'):
        mp.spawn(evaluate,
                args=(args,world_size),
                nprocs=world_size,
                join=True)



    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)
    
    
    tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])

    info_list = []
    for i in range(world_size):
        with open(args.work_dir+f'/instruction_eval_info_list_{i}.json', 'r', encoding='utf-8') as f:
            list_i =  json.load(f)
        info_list += list_i

    generate_text = [entry["generate_text"] for entry in info_list]

    print("calculate BLEU4...")    

    if os.path.exists(f'test_instruction_dataset.json'):
        with open(f'test_instruction_dataset.json', 'r', encoding='utf-8') as f:
            examples_list =  json.load(f)
    

    instruction_inference_results = []
    bleu4_list = []
    for gen_text, example in zip(generate_text, examples_list):

        ans_text = example["answer"]
    
        gen_text = tokenizer.decode(gen_text, skip_special_tokens=True)

        gen_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        bleu4 = sentence_bleu([answer_ids], gen_ids, weights=(0.25, 0.25, 0.25, 0.25))


        example["generate"] = gen_text
        example["bleu4"] = bleu4
        instruction_inference_results.append(example)
        bleu4_list.append(bleu4)

    avg_bleu4 = np.mean(bleu4_list)
    print(f"avg_bleu4:{avg_bleu4}")
    with open(args.work_dir+f'/instruction_brief_eval_info.json', 'w', encoding='utf-8') as f:
        json.dump(f"avg_bleu4:{avg_bleu4}", f, ensure_ascii=False)

    with open(args.work_dir+f'/instruction_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(instruction_inference_results, f, ensure_ascii=False, indent=4)  


"""
python ./evaluator.py --work_dir CompressLLM --batch_size 1

CUDA_VISIBLE_DEVICES=0,1 HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 python ./evaluator.py --work_dir CompressLLM --batch_size 1

Todo: multi gpu parallel inference
"""