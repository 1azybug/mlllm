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
        compress_loss_values = [entry['training_loss']["compress_loss"] for entry in info_list]
        lm_loss_values = [entry['training_loss']["lm_loss"] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]
        
        plt.figure(figsize=(10, 5))
        plt.plot(step_values, ae_loss_values, label="ae_loss")
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


    def evaluate(self):
        rank = 0 # gpu_device_id
        training_config = self.config["training_config"]
        task_config = self.config["task_config"]
        train_examples, eval_examples = get_examples(**self.config["data_config"])
        dataset = get_dataset(task_config["task_type"], eval_examples, batch_size=self.batch_size)
        loader = DataLoader(dataset, batch_size=None)
        
        model = get_model(training_config["model_id"], task_config, rank)
        model = load_adapter(model, save_path_and_name=args.work_dir+'/adapter.pt', log=True)
        model.eval()
        
        info_list=[]
        with torch.no_grad():
            for inputs in tqdm(loader,total=1000):
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
                info_list.append(output["loss_info"])
        
        ae_loss_values = [entry["ae_loss"] for entry in info_list]
        compress_loss_values = [entry["compress_loss"] for entry in info_list]
        lm_loss_values = [entry["lm_loss"] for entry in info_list]
        bleu4_values = [entry["bleu4"] for entry in info_list]
        
        avg_ae_loss = np.mean(ae_loss_values)
        avg_compress_loss = np.mean(compress_loss_values)
        avg_lm_loss = np.mean(lm_loss_values)
        avg_bleu4 = np.mean(bleu4_values)
        logging.info(f"avg_ae_loss:{avg_ae_loss}, avg_compress_loss:{avg_compress_loss}, avg_lm_loss:{avg_lm_loss}, avg_bleu4:{avg_bleu4}")
        
        
        
                
    def run(self):
        # draw training loss
        self.draw_loss()
        self.evaluate()



if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.work_dir+'/evaluate_info.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    evaluator = Evaluator(config, args.work_dir, args.batch_size)
    evaluator.run()


"""
python ./evaluator.py --work_dir CompressLLM --batch_size 1

Todo: multi gpu parallel inference
"""