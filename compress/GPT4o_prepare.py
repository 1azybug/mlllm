import openai
import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir1', type=str, required=True, help='Directory including the instruction_inference_results.json file')
    parser.add_argument('--work_dir2', type=str, required=True, help='Directory including the instruction_inference_results.json file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory storing result')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.work_dir1+f'/instruction_inference_results.json', 'r', encoding='utf-8') as f:
        answer1_list =  json.load(f)

    with open(args.work_dir2+f'/instruction_inference_results.json', 'r', encoding='utf-8') as f:
        answer2_list =  json.load(f)

    with open('eval_prompt.txt', 'r', encoding='utf-8') as f:
        template = f.read()  # 这样会读取整个文件内容为字符串

    eval_cache = []
    index=0
    for answer1, answer2 in zip(answer1_list,answer2_list):
        if index < len(eval_cache):
            continue

        text = answer1["input"]
        question = answer1["prompt"]
        ans1 = answer1["generate"]
        ans2 = answer2["generate"]

        temp = """```
Text:{}
Prompt:{}
Assistant A:{}
Assistant B:{}
```"""
        if index%2==0:
            prompt = template + temp.format(text,question,ans1,ans2)
        else:
            prompt = template + temp.format(text,question,ans2,ans1)

        
        # [Important] remeber index%2==1 is reverse.
        request = {"custom_id": f"request-{index}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-2024-08-06", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}],"max_tokens": 2048}}
        eval_cache.append(request)
        index+=1

    chunk_size = 600
    begin_index = 0
    for i in range(begin_index,len(eval_cache),chunk_size):
        with open(args.output_dir+f"/{args.work_dir1}_vs_{args.work_dir2.replace('&', '_and_')}_GPT4_request{i}-{min(i+chunk_size-1,len(eval_cache)-1)}.jsonl", 'w', encoding='utf-8') as f:
            for item in eval_cache[i:i+chunk_size]:     
                json_string = json.dumps(item, ensure_ascii=False)
                f.write(json_string + '\n')


# python GPT4o_prepare.py --work_dir1 compressLLM_len-510_ratio-15 --work_dir2 compressLLM_len-510-ratio-15_wo-cmp --output_dir GPT4_eval_result/full_vs_wo-cmp
# python GPT4o_prepare.py --work_dir1 compressLLM_len-510-ratio-15_wo-cmp --work_dir2 'compressLLM_len-510-ratio-15_wo-cmp&pe'  --output_dir GPT4_eval_result/wo-cmp_vs_wo-cmp-and-pe
# python GPT4o_prepare.py --work_dir1 compressLLM_len-510-ratio-15_wo-ae --work_dir2 compressLLM_len-510-ratio-15_wo-cmp  --output_dir GPT4_eval_result/wo-ae_vs_wo-cmp