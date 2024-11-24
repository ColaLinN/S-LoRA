## a100-40g S4
# slora


# nsys profile --duration 600 -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cudabacktrace=true -x true --stats=true --force-overwrite true -o nsys/original_adapter_s4 python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/original_adapter_s4.txt
python ../../run_exp.py  --backend slora  --suite a100-40-num-adapter  --model-setting S4  --mode synthetic --output original_adapter_s4.jsonl | tee output/original_adapter_s4.txt
