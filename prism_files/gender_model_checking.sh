# Gender and Sendiment Analysis
echo "BERT" >> gender_results.log
storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> gender_results.log
storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [G gender_sentence=0]" >> gender_results.log
storm --prism bert-base-uncased_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F (gender_sentence<0 & gender_sentence!=-14124)]" >> gender_results.log

echo "GEMMA-2B" >> gender_results.log
storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> gender_results.log
storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [G gender_sentence=0]" >> gender_results.log
storm --prism google_gemma-2b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F (gender_sentence<0 & gender_sentence!=-14124)]" >> gender_results.log

echo "GEMMA-7B" >> gender_results.log
storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> gender_results.log
storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [G gender_sentence=0]" >> gender_results.log
storm --prism google_gemma-7b-it_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F (gender_sentence<0 & gender_sentence!=-14124)]" >> gender_results.log

echo "LLAMA-7B" >> gender_results.log
storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> gender_results.log
storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [G gender_sentence=0]" >> gender_results.log
storm --prism meta-llama_Llama-2-7b-hf_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F (gender_sentence<0 & gender_sentence!=-14124)]" >> gender_results.log

echo "MISTRAL-7B" >> gender_results.log
storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> gender_results.log
storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [G gender_sentence=0]" >> gender_results.log
storm --prism mistralai_Mistral-7B-v0.1_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F (gender_sentence<0 & gender_sentence!=-14124)]" >> gender_results.log

echo "GENSTRUCT-7B" >> gender_results.log
storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> gender_results.log
storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [G gender_sentence=0]" >> gender_results.log
storm --prism NousResearch_Genstruct-7B_0.8_15_5_cuda_The_player_won_because_.prism --prop "P=? [F (gender_sentence<0 & gender_sentence!=-14124)]" >> gender_results.log