import transformers
from peft import (
    LoraConfig,
)
from datasets import load_dataset
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model
import json

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)
    
    # training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = training_args.lora_target_modules if training_args.lora_target_modules else None
    )
    # print(lora_config)
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    assert training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm, "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    
    memory_size = training_args.fixed_mem_size

    train_file = "mini_train_data.json"
    eval_file = "eval_data.json"

    print("Loading dataset...")

    # dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file}, chunksize=2**35) # streaming can be removed if the dataset is not very large.
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["eval"]

    from datasets import Dataset

    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(eval_file, 'r') as f:
        eval_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    model = ICAE(model_args, training_args, lora_config)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=1, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, batch_size=1, fn_kwargs={"model": model, "mem": MEM_TOKENS})   # don't add lm in the dev set.

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    train_model(model, train_dataset, eval_dataset, training_args, data_collator)

main()