# Compare alpha vs k vs both

# Alpha
#storm --prism google_gemma-2b-it_0.1_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.2_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.3_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.4_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.5_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.6_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.7_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.8_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.9_32000_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log

# k
#storm --prism google_gemma-2b-it_1.0_1_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_2_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_3_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_4_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_5_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_6_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_7_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_8_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_1.0_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log

# Both
#storm --prism google_gemma-2b-it_0.1_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.2_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.3_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.4_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.5_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.6_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.7_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.8_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log
#storm --prism google_gemma-2b-it_0.9_9_3_cuda_The_player_won_because_.prism --prop "P=? [F gender_sentence>0]" >> benchmarking.log


storm --prism google_gemma-2b-it_0.9_32000_1_cpu_Her_name_was_.prism --prop "P=? [F step=1]" >> benchmarking.log
storm --prism google_gemma-2b-it_0.9_9_1_cpu_Her_name_was_.prism --prop "P=? [F step=1]" >> benchmarking.log
storm --prism google_gemma-2b-it_1.0_9_1_cpu_Her_name_was_.prism --prop "P=? [F step=1]" >> benchmarking.log