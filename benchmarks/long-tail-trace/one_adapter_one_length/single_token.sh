# python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/20241118_v1.txt
#dummy-lora-13b-rank-32-0
python ../../run_exp.py --backend slora --suite req_len_latency_baseline_adapter_1 --model-setting S4 --longtail --mode synthetic --output req_len_latency_baseline_adapter_1.jsonl | tee output/req_len_latency_baseline_adapter_1.txt
#dummy-lora-13b-rank-16-0
python ../../run_exp.py --backend slora --suite req_len_latency_baseline_adapter_2 --model-setting S4 --longtail --mode synthetic --output req_len_latency_baseline_adapter_2.jsonl | tee output/req_len_latency_baseline_adapter_2.txt
#dummy-lora-13b-rank-64-0
python ../../run_exp.py --backend slora --suite req_len_latency_baseline_adapter_3 --model-setting S4 --longtail --mode synthetic --output req_len_latency_baseline_adapter_3.jsonl | tee output/req_len_latency_baseline_adapter_3.txt