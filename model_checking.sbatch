#!/bin/bash
# python3.10 -m venv venv
# sinfo to show all partitions
# sbatch SCRIPNAME.sh # Run script
# squeue -u dennis # Check status of job
# 

#SBATCH --account=dennis
#SBATCH --job-name=model_checking
#SBATCH --output=model_checking.log   
#SBATCH --partition=a100q # a100q dgx2q hgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Output is visible in stdout
echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S
module avail storm
module show storm/gcc/1.8.1
echo "============"
module load storm/gcc/1.8.1
#module load storm/gcc/1.7.0
srun storm --version


# Gender and Sendiment Analysis
#srun echo "BERT" >> gender_results.log
#srun storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity>0]" >> gender_results.log
#srun storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity<0]" >> gender_results.log
#srun storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity>0]" >> gender_results.log
#srun storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity<0]" >> gender_results.log

#srun echo "GEMMA-2B" >> gender_results.log
#srun storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity>0]" >> gender_results.log
#srun storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity<0]" >> gender_results.log
#srun storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity>0]" >> gender_results.log
#srun storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity<0]" >> gender_results.log

#srun echo "GEMMA-7B" >> gender_results.log
#srun storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity>0]" >> gender_results.log
#srun storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity<0]" >> gender_results.log
#srun storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity>0]" >> gender_results.log
#srun storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity<0]" >> gender_results.log

#srun echo "LLAMA-7B" >> gender_results.log
#srun storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity>0]" >> gender_results.log
#srun storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity<0]" >> gender_results.log
#srun storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity>0]" >> gender_results.log
#srun storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity<0]" >> gender_results.log

#srun echo "MISTRAL-7B" >> gender_results.log
#srun storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity>0]" >> gender_results.log
#srun storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity<0]" >> gender_results.log
#srun storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity>0]" >> gender_results.log
#srun storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity<0]" >> gender_results.log

#srun echo "Genstruct-7B" >> gender_results.log
#srun storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity>0]" >> gender_results.log
#srun storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0 & polarity<0]" >> gender_results.log
#srun storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity>0]" >> gender_results.log
#srun storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence<0 & polarity<0]" >> gender_results.log



# Lolita for different LLMs
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-13b-hf" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-7b-it" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="bert-base-uncased" --start_sequence="Lolita, light of my life," --top_k=15 --token_length=8 --alpha=0.8


# Our Story
#srun python prism_encoder.py --device="cuda" --llm="mistralai/Mistral-7B-v0.1" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="NousResearch/Genstruct-7B" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-13b-hf" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="meta-llama/Llama-2-7b-hf" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-7b-it" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="google/gemma-2b-it" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8
#srun python prism_encoder.py --device="cuda" --llm="bert-base-uncased" --start_sequence="Our story" --top_k=16 --token_length=25 --alpha=0.8




# Does the text quality changes with different number of top_k values?

