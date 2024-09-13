# python prepare_data.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe'
# python ./trainer.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --port 14522
# python ./evaluator.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --batch_size 1


# python instruction_prepare_data.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe'
# python ./instruction_trainer.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --port 14522
# python ./instruction_evaluator.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --batch_size 1

# python prepare_data.py --work_dir compressLLM_debug
# python ./trainer.py --work_dir compressLLM_debug --port 29500
# python ./evaluator.py --work_dir compressLLM_debug --batch_size 1

# python instruction_prepare_data.py --work_dir compressLLM_debug
# python ./instruction_trainer.py --work_dir compressLLM_debug --port 29500
# python ./instruction_evaluator.py --work_dir compressLLM_debug --batch_size 1


# python vanilla_prepare_data.py --work_dir compressLLM_debug
# python ./vanilla_trainer.py --work_dir compressLLM_debug --port 29500
# python ./vanilla_evaluator.py --work_dir compressLLM_debug --batch_size 1

# bash train.sh