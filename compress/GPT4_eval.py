import openai
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir1', type=str, required=True, help='Directory including the instruction_inference_results.json file')
    parser.add_argument('--work_dir2', type=str, required=True, help='Directory including the instruction_inference_results.json file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory storing result')
    parser.add_argument('--API', type=str, required=True, help='closed-source model api')
    return parser.parse_args()

def get_completion(prompt, model=None,api=None): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    openai.api_key  = api
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# Launch multi-process eval
if __name__ == "__main__":
    args = parse_args()

    with open(args.work_dir1+f'/instruction_inference_results.json', 'r', encoding='utf-8') as f:
        answer1_list =  json.load(f)

    with open(args.work_dir2+f'/instruction_inference_results.json', 'r', encoding='utf-8') as f:
        answer2_list =  json.load(f)

    with open('eval_prompt.txt', 'r', encoding='utf-8') as f:
        template = f.read()  # 这样会读取整个文件内容为字符串

    eval_cache = []
    cache_file = args.output_dir+f'/{args.work_dir1}_vs_{args.work_dir2}_GPT4_eval_cache.json'
    if not os.path.exists(cache_file):
        with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(eval_cache, f, ensure_ascii=False, indent=4) 

    with open(cache_file, 'r', encoding='utf-8') as f:
        eval_cache =  json.load(f)

    for index, answer1, answer2 in enumerate(zip(answer1_list,answer2_list)):
        if index < len(eval_cache):
            continue

        text = answer1["input"]
        question = answer1["prompt"]
        ans1 = answer1["generate"]
        ans2 = answer2["generate"]

        if index%2==0:
            prompt = template.format(text,question,ans1,ans2)
        else:
            prompt = template.format(text,question,ans2,ans1)
        
        response = get_completion(prompt, model="gpt-4",api=None)
        try:
            # 将响应转换为字典
            result = json.loads(response)
            if index%2==1:
                if result["choice"].lower()=='a':
                    result["choice"] = 'B'
                elif result["choice"].lower()=='b':
                    result["choice"] = 'A'
        except json.JSONDecodeError:
            print(f"index:{index}, json decode failed,response:{response}")
            result = {"reason": ".", "choice": "."}
        eval_cache.append(result)
        with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(eval_cache, f, ensure_ascii=False, indent=4)  

    a_cnt=0
    b_cnt=0
    tie_cnt=0
    err_cnt=0
    tot=0
    for result in eval_cache:
        tot+=1
        if result["choice"].lower()=='a':
            a_cnt+=1
        elif result["choice"].lower()=='b':
            b_cnt+=1
        elif result["choice"].lower()=='tie':
            tie_cnt+=1
        else:
            err_cnt+=1

    print(f"win_rate:{a_cnt/tot:.2f}")
    print(f"lose_rate:{b_cnt/tot:.2f}")
    print(f"tie_rate:{tie_cnt/tot:.2f}")
    print(f"err_rate:{err_cnt/tot:.2f}")
    print(f"on_par_rate (win+tie):{(a_cnt+tie_cnt)/tot:.2f}")
 
