conda activate slora
cd benchmarks
nohup python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real  > output.log 2>&1 &