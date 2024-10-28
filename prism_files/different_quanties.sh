# Lolita Llama2-7B and Gemma-7B
#storm --prism google_gemma-2b-it_0.9_5_5_cuda_Lolita,_light_of_my_life,.prism --prop "P=? [F (step=5 & lolita>90)]" >> lolita_results.log
#storm --prism google_gemma-7b-it_0.9_5_5_cuda_Lolita,_light_of_my_life,.prism --prop "P=? [F (step=5 & lolita>90)]" >> lolita_results.log

# Sentiment
storm --prism google_gemma-7b-it_0.9_5_4_cuda_The_exam_was.prism --prop "P=? [F (polarity>=30)]" >> sentiment_results.log
storm --prism meta-llama_Llama-2-7b-hf_0.9_5_4_cpu_The_exam_was.prism --prop "P=? [F (polarity>=30)]" >> sentiment_results.log
storm --prism google_gemma-7b-it_0.9_5_4_cuda_The_exam_was.prism --prop "P=? [F (polarity<=-30 & polarity!=-14124)]" >> sentiment_results.log
storm --prism meta-llama_Llama-2-7b-hf_0.9_5_4_cpu_The_exam_was.prism --prop "P=? [F (polarity<=-30 & polarity!=-14124)]" >> sentiment_results.log


# Text quality
#storm --prism google_gemma-2b-it_0.6_3_10_cuda_Our_story.prism --prop "P=? [G readability>5997]" > text_quality_results.log
#storm --prism meta-llama_Llama-2-7b-hf_0.6_3_10_cuda_Our_story.prism --prop "P=? [G readability>5997]" >> text_quality_results.log
#storm --prism text_qualityalpha1.prism --prop "P=? [(true U (text_quality<90)) U text_quality>90]" >> text_quality_results.log
#storm --prism text_qualityalpha2.prism --prop "P=? [(true U (text_quality<90)) U text_quality>90]" >> text_quality_results.log
#storm --prism text_quality.prism --prop "P=? [G text_quality>90]" >> text_quality_results.log
