# English


# 中文
* 必须吐槽一下。其实更应该在v1上面验证位置编码的有效性，因为v1的架构要更简单，但是v2的代码比v1更完善。


## fast start

在forget([见根目录的readme](../README.md))的环境基础上:
```
pip install peft
pip install flash-attn --no-build-isolation
```

```
python prepare.py
```

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


# 复现

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
# 加入位置编码
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




