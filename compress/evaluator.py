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

import logging
from nltk.translate.bleu_score import sentence_bleu

from torch.nn import DataParallel
import torch.multiprocessing as mp

from prepare_data import get_examples
from modeling import get_model, save_adapter, load_adapter
from dataloader import get_dataset

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
        with open(os.path.join(self.work_dir,"info.json")) as f:
            info_list=json.load(f)

        ae_loss_values = [entry['training_loss']["ae_loss"] for entry in info_list]
        # compress_loss_values = [entry['training_loss']["compress_loss"] for entry in info_list]
        compress_loss_values = [-1 if 'compress_loss' not in entry['training_loss'] else entry['training_loss']["compress_loss"] for entry in info_list]
        lm_loss_values = [entry['training_loss']["lm_loss"] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]
        
        plt.figure(figsize=(10, 5))
        plt.plot(step_values, ae_loss_values, label="ae_loss")
        if compress_loss_values[0]!=-1:
            plt.plot(step_values, compress_loss_values, label="compress_loss")
        plt.plot(step_values, lm_loss_values, label="lm_loss")

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(self.work_dir)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.work_dir, 'training_loss.png'))
        plt.close()


    def draw_ema_loss(self, alpha=0.9):

        def exponential_moving_average(values,alpha):
            ema = [values[0]]  # 初始化EMA的第一个值为原始值的第一个值
            for value in values[1:]:
                ema.append(alpha * value + (1 - alpha) * ema[-1])
            return ema

        with open(os.path.join(self.work_dir,"info.json")) as f:
            info_list=json.load(f)

        ae_loss_values = [entry['training_loss']["ae_loss"] for entry in info_list]
        # compress_loss_values = [entry['training_loss']["compress_loss"] for entry in info_list]
        compress_loss_values = [-1 if 'compress_loss' not in entry['training_loss'] else entry['training_loss']["compress_loss"] for entry in info_list]
        lm_loss_values = [entry['training_loss']["lm_loss"] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]

        
        plt.figure(figsize=(10, 5))
        plt.plot(step_values, exponential_moving_average(ae_loss_values,alpha=alpha), label="ae_loss")
        if compress_loss_values[0]!=-1:
            plt.plot(step_values, exponential_moving_average(compress_loss_values,alpha=alpha), label="compress_loss")
        
        plt.plot(step_values, exponential_moving_average(lm_loss_values,alpha=alpha), label="lm_loss")

        plt.xlabel("step")
        plt.ylabel(f"loss(ema_alpha={alpha})")
        plt.title(f"{self.work_dir}_loss(ema_alpha={alpha})")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.work_dir, 'ema_training_loss.png'))
        plt.close()

    def evaluate(self, rank=0):
        # rank:gpu_device_id
        training_config = self.config["training_config"]
        task_config = self.config["task_config"]
        train_examples, eval_examples = get_examples(**self.config["data_config"])
        eval_examples = eval_examples[:32]
        example_num_per_gpu = len(eval_examples)//torch.cuda.device_count()
        assert example_num_per_gpu*torch.cuda.device_count() == len(eval_examples)
        eval_examples = eval_examples[rank*example_num_per_gpu:(rank+1)*example_num_per_gpu]
        print(f"[INFO] GPU{rank}: eval_examples[{rank*example_num_per_gpu}:{(rank+1)*example_num_per_gpu}]")
        dataset = get_dataset(task_config["task_type"], eval_examples, batch_size=self.batch_size)
        loader = DataLoader(dataset, batch_size=None)
        
        model = get_model(training_config["model_id"], task_config, rank)
        model = load_adapter(model, save_path_and_name=self.work_dir+'/adapter.pt', log=False)
        model.eval()

        info_list=[]
        with torch.no_grad():
            for inputs in tqdm(loader,total=len(eval_examples)//self.batch_size):
                inputs = {key:value.to(rank) for key,value in inputs.items()}
                output = model(inputs=inputs)
                
                
                generate_text = model.ae_inference(inputs,generate_num=task_config["segment_size"])
                # print(inputs['ae_targets'].size())
                # print("teacher forcing generate:", torch.argmax(output["logits"], dim=-1).tolist()) # B,S,V -> B,S
                # print("Auto-regressive generate:", generate_text)
                # print("target:", inputs['ae_targets'].tolist())
                
                bleu4 = sentence_bleu([inputs['ae_targets'].tolist()], generate_text, weights=(0.25, 0.25, 0.25, 0.25))
                # print(f"BLEU-4:",bleu4*100)
                
                output["loss_info"]["bleu4"] = bleu4
                if "compress_loss" not in output["loss_info"]:
                    output["loss_info"]["compress_loss"]=-1

                info_list.append(output["loss_info"])

        with open(self.work_dir+f'/eval_info_list_{rank}.json', 'w', encoding='utf-8') as f:
            json.dump(info_list, f, ensure_ascii=False)
        
        ae_loss_values = [entry["ae_loss"] for entry in info_list]
        compress_loss_values = [entry["compress_loss"] for entry in info_list]
        lm_loss_values = [entry["lm_loss"] for entry in info_list]
        bleu4_values = [entry["bleu4"] for entry in info_list]
        
        avg_ae_loss = np.mean(ae_loss_values)
        avg_compress_loss = np.mean(compress_loss_values)
        avg_lm_loss = np.mean(lm_loss_values)
        avg_bleu4 = np.mean(bleu4_values)
        logging.info(f"avg_ae_loss:{avg_ae_loss}, avg_compress_loss:{avg_compress_loss}, avg_lm_loss:{avg_lm_loss}, avg_bleu4:{avg_bleu4}")
        
        
        
                
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
            logging.FileHandler(args.work_dir+f'/evaluate_info_rank{rank}.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    evaluator = Evaluator(config, args.work_dir, args.batch_size)
    evaluator.run(rank)


# Launch multi-process eval
if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(evaluate,
             args=(args,world_size),
             nprocs=world_size,
             join=True)


    info_list = []
    for i in range(world_size):
        with open(args.work_dir+f'/eval_info_list_{i}.json', 'r', encoding='utf-8') as f:
            list_i =  json.load(f)
        info_list += list_i

    ae_loss_values = [entry["ae_loss"] for entry in info_list]
    compress_loss_values = [entry["compress_loss"] for entry in info_list]
    lm_loss_values = [entry["lm_loss"] for entry in info_list]
    bleu4_values = [entry["bleu4"] for entry in info_list]
    
    avg_ae_loss = np.mean(ae_loss_values)
    avg_compress_loss = np.mean(compress_loss_values)
    avg_lm_loss = np.mean(lm_loss_values)
    avg_bleu4 = np.mean(bleu4_values)
    print(f"avg_ae_loss:{avg_ae_loss}, avg_compress_loss:{avg_compress_loss}, avg_lm_loss:{avg_lm_loss}, avg_bleu4:{avg_bleu4}")

    with open(args.work_dir+f'/brief_eval_info.json', 'w', encoding='utf-8') as f:
        json.dump(f"avg_ae_loss:{avg_ae_loss}, avg_compress_loss:{avg_compress_loss}, avg_lm_loss:{avg_lm_loss}, avg_bleu4:{avg_bleu4}", f, ensure_ascii=False)

"""
python ./evaluator.py --work_dir CompressLLM --batch_size 1

CUDA_VISIBLE_DEVICES=0,1 HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 python ./evaluator.py --work_dir CompressLLM --batch_size 1

Todo: multi gpu parallel inference
"""