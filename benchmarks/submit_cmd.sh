#nscc
qsub -I -l select=1:ngpus=1 -l walltime=06:00:00 -P personal-e1327875 -q normal
export PBS_JOBID=

# path
cd scratch/S-LoRA/benchmarks/
conda activate slora

# conda
conda list
conda activate slora
conda env list
nvcc --version
nvidia-smi


# cuda
module av cuda
module load cuda/11.8.0
module unload cuda

#
nsys profile -y 10  -d 600  --gpu-metrics-device=0 --stats=true --force-overwrite true -o a10g_a100_40g_s2_server_full_slora_output_20241111_v1  python launch_server.py --backend slora --num-adapter 200 --num-token 14000 --model-setting S4 --dummy | tee output/a10g_a100_40g_s2_server_full_slora_output_20241111_v1.txt

#scp
scp -v -r /Users/fenglyulin/guzheng/github/S-LoRA e1327875@aspire2a.nus.edu.sg:/home/users/nus/e1327875

#git
git remote show origin
git remote set-url origin git_url
