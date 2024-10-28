# PCTL Model Checking of Large Language Model Text Generation
When provided with input text, a large language model (LLM) generates the subsequent unit of text, known as a token.
This token is appended to the input, continuing the iterative, autoregressive text generation process.
However, LLMs are prone to faults like biases, hallucinations, and low text quality, raising concerns over their reliability.
As LLMs become increasingly popular in critical applications, such as healthcare or technical documentation, it is crucial to verify their consistency to generate high-quality, error-free text.
Given the vast number of tokens at each stage of text generation, formally verifying every possible output of an LLM across multiple future tokens leads to a combinatorial explosion.
We introduce LLMchecker, a model checking-based formal verification method for verifying probabilistic computation tree logic (PCTL) properties of the $\alpha$-$k$-bounded text generation process.
We empirically show that only a limited number of tokens are typically chosen during text generation, which are not always the same.
This insight drives the creation of $\alpha$-$k$-bounded text generation, narrowing the focus to a small set of maximal $k$ tokens at every step of the text generation process.
Our verification method considers an initial string and the subsequent $n$ tokens while accommodating diverse text quantification methods, such as evaluating text quality and biases.
The threshold $\alpha$ further trims the selected tokens, only choosing those surpassing or meeting it in cumulative probability.
LLMchecker then allows to formally verify the PCTL properties of the $\alpha$-$k$-bounded LLMs.
We demonstrate the applicability of our method across several LLMs, including Llama, Gemma, Mistral, Genstruct, and BERT.
To our knowledge, this is the first time PCTL-based model checking has been used to check the consistency of the LLM text generation process.

## Data
The generated PRISM DTMC models can be found in `prism_files`.
The plots can be found in `figures`.

# Running the code
To run the code, you need to have the following dependencies installed `requirements.txt` and the Storm model checker.
The `top_k_prob_analysis.py` script can be used to generate the analysis of the top-k tokens.
The `prism_encoder.py` script can be used to generate the PRISM DTMC models.
The experiments can be found in `encoding.sbatch`. Comment out the experiments you do want to run (also remove the srun if not running on a cluster).