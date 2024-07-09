python prepare_data.py --work_dir CompressLLM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./trainer.py --work_dir CompressLLM --port 12314