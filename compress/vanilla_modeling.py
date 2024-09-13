# vanilla_*.py modified from instruction_*.py
# only rewrite forward and lm_inference fuction(to be a vanilla llama)

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers

# from peft import prepare_model_for_kbit_training

class LinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=16, weight=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight)
        result += self.scale*(x@self.lora_A@self.lora_B)
        return result
    

class EmbeddingLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, padding_idx, r=128, weight=None):
        super().__init__()
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)
        
        
    def forward(self, x):
        result = F.embedding(x, self.weight, self.padding_idx)
        after_A = F.embedding(x, self.lora_A, self.padding_idx)
        result += self.scale*(after_A@self.lora_B)
        return result

class CompressLLM(torch.nn.Module):
    def __init__(self, model_id, mem_size, head_num, device_rank, task_config):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{device_rank}",
        )
        self.device = f"cuda:{device_rank}"
        self.task_config = task_config
        config = self.model.config
        self.vocab_size = config.vocab_size
        self.mem_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((mem_size, config.hidden_size)), requires_grad=True)
        self.special_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((2, config.hidden_size)), requires_grad=True)
        self.head_num = head_num

        self.compress_head = None

        # self.compress_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, head_num*128, bias=False, device=f"cuda:{device_rank}", dtype=self.model.model.embed_tokens.weight.dtype),
        #     nn.Linear(head_num*128, head_num*config.vocab_size, bias=False, device=f"cuda:{device_rank}", dtype=self.model.model.embed_tokens.weight.dtype)
        #     )
        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.mem_tokens, mean=mean, std=std)
        nn.init.normal_(self.special_tokens, mean=mean, std=std)

    def forward(self,inputs):

        # print(inputs["input_ids"].shape)
        # if inputs['lm_targets'] is not None:
        #     print(inputs['lm_targets'].shape)
        # else:
        #     print(inputs['lm_targets'])

        # if len<=segment_len  all_ids = inputs["input_ids"]
        # if len>=segment_len + 2  all_ids = torch.cat([inputs["input_ids"], inputs['lm_targets']], dim=-1)
        # if len==segment_len +1 all_ids = inputs["input_ids"]+<eos>  (but inputs['lm_targets'] is None)
        if inputs['lm_targets'] is not None:
            all_ids = torch.cat([inputs["input_ids"], inputs['lm_targets']], dim=-1)
        else:
            all_ids = inputs["input_ids"]
        if all_ids.shape[-1] != inputs["instruction_target"].shape[-1]: # if len==segment_len +1 they will be equal
            all_ids = all_ids[:,:-1]
        assert all_ids.shape[-1] == inputs["instruction_target"].shape[-1]

        # print(inputs)
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        inputs_embeds = self.model.model.embed_tokens(all_ids)

        outputs = self.model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
            )
        original_logits = outputs.logits

        loss_info = {}

        logits = original_logits.contiguous().view(-1, self.vocab_size)
        inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)

        lm_loss = self.loss_fct(logits, inputs["instruction_target"])
        loss_info["lm_loss"] = lm_loss.item()
        return {"loss":lm_loss, "loss_info":loss_info}


    def lm_inference(self,inputs,segment_size):

        if inputs['lm_targets'] is not None:
            all_ids = torch.cat([inputs["input_ids"], inputs['lm_targets']], dim=-1)
        else:
            all_ids = inputs["input_ids"]


        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        inputs_embeds = self.model.model.embed_tokens(all_ids)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = inputs_embeds.clone()
        
        for i in range(4096):

            out = self.model(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
              
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)

            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(inputs_embeds.device)
            generate_text.append(next_token_id.item())
            if next_token_id.item() == 2: # eos
                return generate_text

        return generate_text



def save_adapter(model,save_path_and_name='adapter.pt', log=False):
    adapter_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if log:
                print("[Save Adapter]",name)
            adapter_name.add(name)
            
    state_dict = model.state_dict()
    adapter_state_dict = {name: param for name, param in state_dict.items() if name in adapter_name}
    torch.save(adapter_state_dict, save_path_and_name)

def load_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # 先加载到CPU
    if log:
        print("Loading adapter parameters:")
        for name, _ in adapter_state_dict.items():
            print(f"[Load Adapter] {name}")
    
    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}
    
    model.load_state_dict(adapter_state_dict, strict=False)
    return model



def get_model_for_compress(model_id, task_config, rank):

    def add_compress_lora(model, task_config):
        for name, module in model.named_children():
            
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name, LinearLoraLayer(module.in_features, module.out_features, weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx,weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    

    # config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = CompressLLM(
        model_id,
        mem_size=task_config["mem_size"],
        head_num=task_config["head_num"],
        device_rank=rank,
        task_config=task_config
    )
    # model = prepare_model_for_kbit_training(model)
    add_compress_lora(model, task_config)
    return model


def get_model(model_id, task_config, rank):
    if task_config["task_type"] == "Compress":
        return get_model_for_compress(model_id, task_config, rank)
    raise Exception("Don't exist [{task_type}] task.")



def load_model_with_adapter(model_id, task_config, rank, save_path_and_name='adapter.pt', log=False):
    model = get_model(model_id, task_config, rank)
    load_adapter(model, save_path_and_name, log)
    return model
# python /home/liuxinyu/zrs/forget-me-not/models/llama3.py