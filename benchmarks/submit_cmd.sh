#nscc
qsub -I -l select=1:ngpus=1 -l walltime=03:00:00 -P personal-e1327875 -q normal
export PBS_JOBID=

# path
cd scratch/S-LoRA/benchmarks/ && conda activate slora

# conda
conda list
conda activate slora
conda env list
nvcc --version
nvidia-smi


#cuda
module av cuda
module load cuda/11.8.0
module load cuda/12.2.2
module unload cuda

#python
# set environment variables:
export TORCH_CUDA_ARCH_LIST="8.0"
# install dependencies
pip install torch==2.0.1
pip install -e .

# # others
# nsys profile -y 10  -d 600  --gpu-metrics-device=0 --stats=true --force-overwrite true -o a10g_a100_40g_s2_server_full_slora_output_20241111_v1  python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/a10g_a100_40g_s2_server_full_slora_output_20241111_v1.txt
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -o my_profile python nsys_exp.py
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o my_profile python nsys_exp.py
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop-shutdown --cudabacktrace=true -x true -o long_tail_20241118_v1 python nsys_exp.py
# # x, longtail
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop-shutdown --cudabacktrace=true -x true -o long_tail_20241118_v1       python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/20241118_v1.txt
# nsys launch  python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/a10g_a100_40g_s2_server_full_slora_output_long_tail_20241118_v1.txt
# nsys start -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --cudabacktrace=true -x true -o long_tail_20241118_v1
# √, auto terminate longtail after a while
nsys profile --duration 300 -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cudabacktrace=true -x true --stats=true --force-overwrite true -o nsys/long_tail_20241118_v1  python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/20241118_v1.txt
nsys profile --duration 120 -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cudabacktrace=true -x true --stats=true --force-overwrite true -o nsys/long_tail_20241118_v2  python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/20241118_v1.txt

#server
python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/20241118_v1.txt
python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/20241118_v1.txt

#req
python run_exp.py --backend slora --suite a100-40-num-adapter-short --model-setting S4 --mode synthetic --output 20241118_v1.jsonl | tee output/20241118_v1.txt

#scp
# recursively copy from
scp source target
scp -v -r e1327875@aspire2a.nus.edu.sg:/home/users/nus/e1327875/scratch/S-LoRA/benchmarks/my_profile.nsys-rep /Users/fenglyulin/Downloads
scp -v -r /Users/fenglyulin/guzheng/github/S-LoRA e1327875@aspire2a.nus.edu.sg:/home/users/nus/e1327875

#git
git remote show origin
git remote set-url origin git_url
