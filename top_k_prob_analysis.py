import pandas as pd
from transformer_model import *
import random
import matplotlib.pyplot as plt
import argparse
import torch
import gc


def get_random_n_words(sentence, n):
    return " ".join(sentence.split(" ")[0:n])

def get_args():
    parser = argparse.ArgumentParser(description="Model checking for PRISM models")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cpu)", default="cpu")
    parser.add_argument("--sampe_size", type=int, help="Number of samples to take.", default=2)
    parser.add_argument("--max_top_k", type=int, help="Max top k tokens to check.", default=2)
    parser.add_argument("--df_size", type=int, help="Max top k tokens to check.", default=2)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    SAMPLE_SIZE = args.sampe_size
    MAX_TOP_K = args.max_top_k
    
    df = pd.read_csv('data/chatgpt_prompts.csv')
    df = df[0:args.df_size]
    print(df.head())
    all_llms = {}
   
    # prompt str column to continous string variable
    # "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf"
    for llm in ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.1", "NousResearch/Genstruct-7B", "meta-llama/Llama-2-7b-hf", "google/gemma-7b-it", "google/gemma-2b-it", "bert-base-uncased"]:
        model = TModel(llm, args.device)
        all_results = {}
        for top_k in range(1,MAX_TOP_K+1):
            prob_sum = 0
            for idx, row in df.iterrows():
                for i in range(SAMPLE_SIZE):
                    print(f"LLM: {llm} Top K: {top_k} Data Sample: {idx}/{len(df)} Sample: {i}")
                    current_input_text = row['prompt'].replace('\n', ' ')
                    # Get random n words
                    current_input_text = get_random_n_words(current_input_text, random.randint(0, len(current_input_text.split(" "))))
                    # Get word prob pairs
                    word_prob_pairs = model.get_next_word_probs(current_input_text, top_k=top_k)
                    prob_sum += model.sum_prob_pairs(word_prob_pairs)

            prob_sum = prob_sum / (SAMPLE_SIZE*len(df))
            all_results[f"{top_k}"] = prob_sum
        all_llms[llm] = all_results
        del model.model
        del model.tokenizer
        del model
        # Free memory
        gc.collect()
        # Free GPU
        torch.cuda.empty_cache()

    # Save all_llms to csv
    all_llms_df = pd.DataFrame(all_llms)
    all_llms_df.to_csv(f"figures/top_{MAX_TOP_K}_{SAMPLE_SIZE}_prob_analysis.csv")
    print(all_llms_df.head())

    # Plot results as line plot
    for llm, results in all_llms.items():
        if llm != "bert-base-uncased":
            llm = llm.split("/")[1]
        plt.plot(list(results.keys()), list(results.values()), label=llm)
    plt.xlabel('Top k tokens')
    plt.ylabel('Probability')
    plt.grid()
    # Place legend to the right outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    # Make sure that the legend is not cut off
    plt.tight_layout()

    plt.savefig(f'figures/top_{MAX_TOP_K}_{SAMPLE_SIZE}_prob_analysis.png')
    plt.savefig(f'figures/top_{MAX_TOP_K}_{SAMPLE_SIZE}_prob_analysis.pgf')

    
    
    

    
