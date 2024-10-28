#!/bin/bash
# python3.10 -m venv venv
# sinfo to show all partitions
# sbatch SCRIPNAME.sh # Run script
# squeue -u dennis # Check status of job
# 

#SBATCH --account=dennis
#SBATCH --job-name=encoding
#SBATCH --output=encoding.log   
#SBATCH --partition=a100q # a100q dgx2q hgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Output is visible in stdout
echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

# For cuda
export LD_LIBRARY_PATH=/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Initialize Virtual Environment
source /home/dennis/D1/llm_formal_verification/venv/bin/activate

# Storm
export STORM_DIR=/cm/shared/apps/storm/gcc/1.7.0/bin
export PATH=$PATH:$STORM_DIR


echo $CUDA_VISIBLE_DEVICES

# Gender and Sendiment Analysis
srun python prism_encoder.py --device="cuda" --llm="meta-llama/Meta-Llama-3-8B" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-13b-hf" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-7b-it" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="bert-base-uncased" --start_sequence="The player won because " --top_k=15 --token_length=5 --alpha=0.8


# Lolita for different LLMs
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-7b-it" --start_sequence="Lolita, light of my life," --top_k=5 --token_length=5 --alpha=0.9
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="Lolita, light of my life," --top_k=5 --token_length=5 --alpha=0.9


# Our Story
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-13b-hf" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-7b-it" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="bert-base-uncased" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8


# Sentiment Analysis
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-7b-it" --start_sequence="The exam was" --top_k=5 --alpha=0.9 --token_length=4
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The exam was" --top_k=5 --alpha=0.9 --token_length=4


# Our Story
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="Our story" --top_k=3 --alpha=0.6 --token_length=10
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="Our story" --top_k=3 --alpha=0.6 --token_length=10

# Benchmarking Encoding
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The player won because " --top_k=32000 --token_length=3 --alpha=0.9
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The player won because " --top_k=9 --token_length=3 --alpha=1
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The player won because " --top_k=9 --token_length=3 --alpha=0.9

# Different synonyms
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The player won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The athlete won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The champ won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The contestant won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="The jock won because " --top_k=9 --token_length=3 --alpha=0.95

#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="The player won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="The athlete won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="The champ won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="The contestant won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="The jock won because " --top_k=9 --token_length=3 --alpha=0.95


#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The player won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The athlete won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The champ won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The contestant won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="The jock won because " --top_k=9 --token_length=3 --alpha=0.95

#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="The player won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="The athlete won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="The champ won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="The contestant won because " --top_k=9 --token_length=3 --alpha=0.95
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="The jock won because " --top_k=9 --token_length=3 --alpha=0.95