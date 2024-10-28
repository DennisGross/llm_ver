#!/bin/bash
# python3.10 -m venv venv
# sinfo to show all partitions
# sbatch SCRIPNAME.sh # Run script
# squeue -u dennis # Check status of job
# 

#SBATCH --account=dennis
#SBATCH --job-name=top_k
#SBATCH --output=top_k.log   
#SBATCH --partition=dgx2q # a100q dgx2q hgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Output is visible in stdout
echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

# For cuda
export LD_LIBRARY_PATH=/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Initialize Virtual Environment
source /home/dennis/D1/llm_formal_verification/venv/bin/activate


echo $CUDA_VISIBLE_DEVICES
srun python top_k_prob_analysis.py --device=cuda --sampe_size=10 --max_top_k=100 --df_size=10000000
