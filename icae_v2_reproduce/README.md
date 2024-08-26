# English


# 中文
* 必须吐槽一下。其实更应该在v1上面验证位置编码的有效性，因为v1的架构要更简单，但是v2的代码比v1更完善。


## fast start

```
cd mlllm/icae_v2_reproduce
```

在forget([见根目录的readme](../README.md))的环境基础上:
```
pip install peft
pip install flash-attn --no-build-isolation
```

```
python prepare.py
```

如果compress文件夹下没有long_text.json文件，则
```
cd ../compress
python prepare_data.py --work_dir compressLLM_len-510_ratio-15
cd ../icae_v2_reproduce
```

登录wandb
```
wandb login
enter your API
```
因为这里只测试Positon id对收敛的好处，所有只看wandb上的eval图就行。（代码里eval的是AE task）

使用默认的position id
```
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=8 pretrain.py \
--output_dir <一个空文件夹，用来保存训练结果> \
--model_name_or_path <你的模型路径> \
--model_max_length 512 \
--lm_ratio 0.5 \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 1 \
--num_train_epochs 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--warmup_steps 300 \
--max_grad_norm 2.0 \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy "steps" \
--eval_steps 100 \
--add_special_token_for_lm True \
--use_my_pe False

```

使用我们的position id
batch size必须为1；否则报错
```
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=8 pretrain.py \
--output_dir <一个空文件夹，用来保存训练结果> \
--model_name_or_path <你的模型路径> \
--model_max_length 512 \
--lm_ratio 0.5 \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 1 \
--num_train_epochs 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--warmup_steps 300 \
--max_grad_norm 2.0 \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy "steps" \
--eval_steps 100 \
--add_special_token_for_lm True \
--use_my_pe True
```

使用优化all-liear(不包括embedding和lm head)的Lora config
```
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=8 pretrain.py \
--output_dir <一个空文件夹，用来保存训练结果> \
--model_name_or_path <你的模型路径> \
--model_max_length 512 \
--lm_ratio 0.5 \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 1 \
--num_train_epochs 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--warmup_steps 300 \
--max_grad_norm 2.0 \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy "steps" \
--eval_steps 100 \
--add_special_token_for_lm True \
--use_my_pe False
--lora_target_modules 'all-linear'
```

使用my position id和优化all-liear(不包括embedding和lm head)的Lora config
```
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=8 pretrain.py \
--output_dir <一个空文件夹，用来保存训练结果> \
--model_name_or_path <你的模型路径> \
--model_max_length 512 \
--lm_ratio 0.5 \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 1 \
--num_train_epochs 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--warmup_steps 300 \
--max_grad_norm 2.0 \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy "steps" \
--eval_steps 100 \
--add_special_token_for_lm True \
--use_my_pe True
--lora_target_modules 'all-linear'
```

#### torchrun
accelerate launch不行,试试torchrun
```
# H20卡好像有bug.
# 可以加入
from datasets import disable_caching
disable_caching()
```

H20再不行就用一张卡跑吧，多等几天

```
MIXED_PRECISION="bf16" torchrun --nproc_per_node=8 --nnodes=1 pretrain.py \
--output_dir <一个空文件夹，用来保存训练结果> \
--model_name_or_path <你的模型路径> \
--model_max_length 512 \
--lm_ratio 0.5 \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 1 \
--num_train_epochs 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--warmup_steps 300 \
--max_grad_norm 2.0 \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy "steps" \
--eval_steps 100 \
--add_special_token_for_lm True \
--use_my_pe True
```

## 复现

#### step1
从 https://github.com/getao/icae/tree/main/code/icae_v2 中将pretrain.py、modeling_icae_multi_span.py、training_utils.py复制到当前目录下


#### step2
在forget([见根目录的readme](../README.md))的环境基础上:
```
pip install peft
pip install flash-attn --no-build-isolation
```

#### step3
~~icae_v2默认设置压缩率为4，mem_token数量为128，总长为512。~~
~~因此总长<=512可以让v2变成v1。~~

生成数据，这里仍使用DKYoon/SlimPajama-6B。
注意：这里使用了compress项目下生成的long_text.json。
（long_text.json由/compress/prepare_data.py生成）
```
python prepare.py
```
每条数据的字符串长度不超过4096（char）

在training_utils.py第61行加入
```
    input_ids = input_ids[:length]
```
在modeling_icae_multi_span.py第179行加入:
```
        assert num_segments==1, num_segments
```
#### step4
将pretrain.py第33,34行改成
```
    train_file = "train_data.json"
    eval_file = "eval_data.json"
```
将pretrain.py第38行到41行改成
```
    from datasets import Dataset

    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(eval_file, 'r') as f:
        eval_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

```
否则会报错
```
ValueError: The train_dataset does not implement __len__, max_steps has to be specified. The number of steps needs to be known in advance for the learning rate scheduler.
```
#### step5
根据 https://github.com/getao/icae/issues/16#issuecomment-2103595823 batch_size只能为1 (触发的bug是mem_token的维度是[1,128],即mem_token的batch size没有扩展到batch size维度) ~~我的代码也只支持batch_size为1，也不好吐槽别人~~

pretrain.py ,57行、58行将batch_size设置为1

除了batch size，其他超参数参考 https://arxiv.org/pdf/2307.06945 的附录A
```
CUDA_VISIBLE_DEVICES=7 python pretrain.py --output_dir ../../icae_output \
--model_name_or_path <模型> \
--model_max_length 512 \
--lm_ratio 0.5 \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 1 \
--num_train_epochs 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--warmup_steps 300 \
--max_grad_norm 2.0 
```
## 加入位置编码
为了控制消融:

在modeling_icae_multi_span.py 77行加入
```
    use_my_pe: bool = field(
        default=False,
        metadata={"help": "use my position_id instead of default position_id of llama"},
    )

```
在modeling_icae_multi_span.py 141行加入
```
        self.mem_position_ids = torch.arange((self.mean_compression_rate+1)/2, self.mean_compression_rate*self.mem_size+1, step=self.mean_compression_rate, device=device).unsqueeze(0)

```
在modeling_icae_multi_span.py 183行加入
```

        # print('prompt_answer_ids:', prompt_answer_ids.shape)
        # print('ae_token:', self.ae_token_id)
        # print('lm_token:', self.lm_token_id)
        # print('special_token:', prompt_answer_ids[0,self.mem_size])
        is_ae_task = (self.ae_token_id == prompt_answer_ids[0,self.mem_size].item())
        # print(f'is_ae_task:{is_ae_task}')

        input_position_ids = torch.arange(1,segment_length+1,device=prompt_answer_embs.device).unsqueeze(0)
        encode_position_ids = torch.cat([input_position_ids,self.mem_position_ids],dim=1)

        second_segment_len = prompt_answer_ids.size(1)-self.mem_size 
        latter_position_ids = torch.arange(0,0+second_segment_len,device=prompt_answer_embs.device).unsqueeze(0)
        ae_decode_position_ids = torch.cat([self.mem_position_ids,latter_position_ids],dim=1)

        latter_position_ids = torch.arange(segment_length,segment_length+second_segment_len,device=prompt_answer_embs.device).unsqueeze(0)
        lm_decode_position_ids = torch.cat([self.mem_position_ids,latter_position_ids],dim=1)

        decode_position_ids = ae_decode_position_ids if is_ae_task else lm_decode_position_ids
        # print(f'lm_decode_position_ids:{lm_decode_position_ids}; ae_decode_position_ids:{ae_decode_position_ids}')
```


在modeling_icae_multi_span.py 214行改成
```
            if self.training_args.use_my_pe:
                segment_compress_outputs = self.icae(position_ids=encode_position_ids, inputs_embeds=segment_input_embedding, output_hidden_states=True)               
            else:
                segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
```

在modeling_icae_multi_span.py 234行改成
```
            if self.training_args.use_my_pe: 
                decoder_outputs = self.decoder(position_ids=decode_position_ids, inputs_embeds=prompt_answer_embs, output_hidden_states=True)             
            else:
                decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
```
在modeling_icae_multi_span.py 240行改成
```
                if self.training_args.use_my_pe: 
                    decoder_outputs = self.icae(position_ids=decode_position_ids, inputs_embeds=prompt_answer_embs, output_hidden_states=True)             
                else:
                    decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
```



## 加入Lora_config控制

只在q,v上加Lora可能无法发挥position id的作用（rope影响q,k），而且这个设置也不是一个好的设置。
改成all_linear应该会更好
在modeling_icae_multi_span.py 81行加入
```
    lora_target_modules: str = field(
        default="",
        metadata={"help": "lora_target_modules of lora config; q_proj, v_proj default in Llama(see: https://github.com/huggingface/peft/blob/670d0fac316d4618bff6f5502793225a10888801/src/peft/utils/constants.py#L96)"},
    )

```

在pretrain.py24行后面插入
```
        target_modules = training_args.lora_target_modules if training_args.lora_target_modules else None
```
保险起见,在pretrain.py27行插入
```
    print(lora_config)
```

添加以下参数会获得更好的效果 ~~大概~~
```
--lora_target_modules 'all-linear'
```

## 位置编码小更新
在modeling_icae_multi_span.py 145行改成 (加了取整，虽然rope可以不是整数，但预训练中模型只见过整数)
```
        self.mem_position_ids = torch.arange((self.mean_compression_rate+1)//2, self.mean_compression_rate*self.mem_size+1, step=self.mean_compression_rate, device=device).unsqueeze(0)
```

感觉这个位置编码的方法不适合icae，因为当被压缩的序列小于mean_compression_rate*self.mem_size时，大部分mem token的位置会离得比较远。

## 位置编码再更新
注释掉
```
        # self.mem_position_ids = torch.arange((self.mean_compression_rate+1)//2, self.mean_compression_rate*self.mem_size+1, step=self.mean_compression_rate, device=device).unsqueeze(0)
```

modeling_icae_multi_span.py 195行插入
```
        compression_rate = segment_length/self.mem_size
        # 完全的范围是[0.5, segment_length+0.5), 该数轴的总长为segment_length, 中间是[1,segment_len]
        mem_position_ids = torch.arange((compression_rate+1)/2, segment_length+0.5, step=compression_rate, device=device).unsqueeze(0)
        mem_position_ids = torch.round(mem_position_ids)
```
[1,segment_length]包含segment_length个整数，但是长度却只有segment_length-1

左右各加0.5使长度为segment_length

将segment_length分成self.mem_size份，每份长度为compression_rate

第一份的中点为0.5+compression_rate/2，下一份中点到上一份中点的距离为compression_rate，最后一份的中点必然小于右端点segment_length+0.5，为segment_length+0.5-compression_rate/2。

## 自闭
我为什么要复现这个,很多实验设置都不一样。有这时间我再检查七八遍我的代码不好吗?
目前还有一个导致我的position id不佳的原因是decoder没有训练。

## 让decoder参与训练
具体修改过程不写了，好奇的话直接看commit:fire decoder and disable gradient_checkpointing。
