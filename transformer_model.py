import torch
from transformers import AutoModelForCausalLM , AutoTokenizer, AutoModel, LlamaForCausalLM
from scipy.spatial.distance import cosine
import numpy as np

REMAINING_PLACEHOLDER = "THIS IS A TERMINAL STATE!!!"

class PlagiarismChecker:

    def __init__(self, model='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def semantic_similarity_plagiarism_check(self, text1, text2, threshold=0.7):
        """
        Checks if text2 is a plagiarism of text1 based on semantic similarity using BERT embeddings.
        
        Parameters:
        - text1, text2: Input texts to compare.
        - model_name: Pre-trained BERT model to use.
        - threshold: Similarity threshold to consider the text as plagiarized.
        
        Returns:
        - similarity: Semantic similarity between the two texts.
        - is_plagiarized: Boolean indicating potential plagiarism.
        """
        # Encode and compute embeddings
        with torch.no_grad():
            inputs1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs1 = self.model(**inputs1).last_hidden_state.mean(dim=1)
            outputs2 = self.model(**inputs2).last_hidden_state.mean(dim=1)
        
        # Compute cosine similarity
        similarity = 1 - cosine(outputs1[0].numpy(), outputs2[0].numpy())
        
        # Determine plagiarism
        is_plagiarized = similarity >= threshold
        
        return similarity, is_plagiarized


class TModel:

    def __init__(self, model="google/gemma-2b-it", device="cuda"):
        # "meta-llama/Llama-2-7b-hf"
        # google/gemma-2b-it
        # bert-base-uncased
        # Initialize the model and the tokenizer.
        self.device = device
        self.model_name = model.replace("/", "_")
        access_token = "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"
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=access_token, trust_remote_code=True)
        #model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model, token=access_token).to(self.device)


    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions

    def get_next_n_words(self, sentence, do_sample=True, n = 1):
        input_ids = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_new_tokens=n, do_sample=do_sample)
        return self.tokenizer.decode(outputs[0])
    
    def get_alpha_words(self, word_prob_pairs, alpha, top_k):
        # Get the alpha words.
        alpha_words = []
        alpha_sum = 0
        k = 0
        for word, prob in word_prob_pairs:
            alpha_sum += prob
            k +=1
            alpha_words.append((word, prob))
            if alpha_sum >= alpha or k >= top_k:
                break
        return alpha_words, 1-alpha_sum

    def get_next_word_probs(self, sentence, top_k):
        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)
        
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()

        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))
    
    def softmax_word_prob_pairs(self, pairs):
        # Get the softmax probabilities for the word probabilities.
        words, probs = zip(*pairs)
        softmax_probs = np.exp(probs) / np.sum(np.exp(probs))
        return list(zip(words, softmax_probs))
    
    def sum_prob_pairs(self, pairs):
        # Get the sum of the word probabilities.
        return sum([prob for _, prob in pairs])

    def get_prob_for_word(self, pairs, word, case_sensitive=False):
        for pair_word, prob in pairs:
            if case_sensitive==False:
                if str(pair_word).lower() == str(word).lower():
                    return prob
            else:
                if str(pair_word) == str(word):
                    return prob
        return None

    def get_word_prob_with_max_prob(self, pairs):
        max_prob = -1
        max_prob_word = ""
        for pair_word, prob in pairs:
            if prob > max_prob:
                max_prob = prob
                max_prob_word = pair_word
        return max_prob_word, max_prob

    def get_word_prob_with_min_prob(self, pairs):
        min_prob = 2
        min_prob_word = ""
        for pair_word, prob in pairs:
            if prob < min_prob:
                min_prob = prob
                min_prob_word = pair_word
        return min_prob_word, min_prob
    
    def get_words_seq_prob(self, input_str, words, top_k=100000):
        current_sentence = input_str
        current_prob = 1
        for idx, word in enumerate(words):
            predictions = self.get_predictions(current_sentence)
            tuples = self.get_next_word_probs(current_sentence, top_k)
            # Get the probability of the word.
            prob = self.get_prob_for_word(tuples, word)
            if prob is not None:
                current_prob *= self.get_prob_for_word(tuples, word)
            current_sentence = self.concatenate_word_to_sentence(current_sentence, word)
            
        return current_prob, current_sentence
    
    @staticmethod
    def concatenate_word_to_sentence(sentence, word):
        # Concatenate the word to the sentence.
        # Handle if alphanumeric or punctuation.
        if word.isalnum():
            if len(word) == 1:
                if "m" == word:
                    return sentence + word
            return sentence + " " + word
        else:
            return sentence + word
        
    @staticmethod
    def clean_tokenized_list(tokenized_list):
        for i in range(len(tokenized_list)):
            if tokenized_list[i].startswith("▁"):
                tokenized_list[i] = tokenized_list[i][1:]
        return tokenized_list
    
    @staticmethod
    def sentence_to_tokenized_list(sentence, tokenizer):
        return tokenizer.tokenize(sentence)
    
    def tokenized_list_to_sentence(self,tokenized_list):
        sentence = ""
        if self.model_name == "bert-base-uncased":
            for i in range(len(tokenized_list)):
                if i == 0:
                    sentence += tokenized_list[i]
                else:
                    if len(tokenized_list[i]) > 1 and tokenized_list[i][0].isalnum():
                        sentence += " " + tokenized_list[i]
                    else:
                        sentence += tokenized_list[i]
            return sentence
        for i in range(len(tokenized_list)):
            if tokenized_list[i].startswith("▁"):
                sentence += " "
                sentence += tokenized_list[i][1:]
            else:
                sentence += tokenized_list[i]
        return sentence



if "__main__" == __name__:
    model = TModel(model="google/gemma-2b-it", device="cpu")
    #plagiarism_checker = PlagiarismChecker()
    #print(model.tokenizer.tokenize(" table next to the window."))
    #print(model.get_words_seq_prob("The cat is on the", model.tokenizer.tokenize(" table next to.", return_tensors="pt")))
    #print(plagiarism_checker.semantic_similarity_plagiarism_check("World", "World"))
    #token_list = TModel.sentence_to_tokenized_list("The cat is on the table next to the window.", model.tokenizer)
    #print(token_list)
    start_string = "The nurse is going home because"
    #print(model.get_words_seq_prob(start_string, ["he"]))
    #print(model.get_next_word_probs(start_string, 10))
    word_prob_pairs = model.get_next_word_probs(start_string, top_k=10)
    print(word_prob_pairs)
    word_prob_pairs, remaining_prob = model.get_alpha_words(word_prob_pairs, alpha=0.9, top_k=10)
    print(word_prob_pairs)
    for word, prob in word_prob_pairs:
        print('=====================')
        print("Word: ", word, " Prob: ", prob)
        # Create new sentence with word
        n_start_sequence = model.concatenate_word_to_sentence(start_string, word)
        print(n_start_sequence)
        print(model.sentence_to_tokenized_list(n_start_sequence, model.tokenizer))
        token_list = model.clean_tokenized_list(model.sentence_to_tokenized_list(n_start_sequence, model.tokenizer))
        print(token_list)
