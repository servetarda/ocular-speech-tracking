# 2. POS tagging and calculations


# Import necessary libraries

# Spact is a library designed for complex NLP tasks like syntactic/morphosyntactic parsing and more
import spacy
# Deep learning framework by Meta for tensor computations and GPU acceleration
# Core tensor operations are computing dot products, matrix multiplications, applying activation functions, automatic differentiation and backpropagation.
import torch
# Provides a collection of functions for common operations on tensors, including activation functions, loss functions, and other utilities.
import torch.nn.functional as F
# Import the Hugging Face Transformers library for model and tokenizer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


## This class is only a definition/recipe of calculations. We are not putting our input here but later in the second part ##

class POSPredictor:
    """
    Computes Part-of-Speech (POS) predictions, surprisal, and entropy
    """
    # "__init__" directly passes when "POSPredictor is called"
    def __init__(self, spacy_model="de_dep_news_trf"):
        # We are basically initiate and implement the model here.
        # this model will handle all types of NLP operations 
        self.nlp = spacy.load(spacy_model)

        # context_text: String representing prior text context.
        # actual_word: Ground-truth word presented in the stimulus.
        # candidate_predictions: Dict of {candidate_word: probability} from top-k/nucleus. 
        # This function is to create POS tags via SpaCy
    def compute_word_pos(self, context_text, actual_word, candidate_predictions):
        # This function strips the special characters 
        clean_word = actual_word.strip(",.?!:;\"'")
        
        # Here concatinate the current word with the prior context for POS tagging
        full_actual_text = f"{context_text} {clean_word}".strip()
        # With the model loaded object and previously generated function, we are letting our model to work on our text composed of prior text and current word
        doc_actual = self.nlp(full_actual_text)
        # We extract universal POS tag of the target word (doc_actual[-1]) after it has been parsed with "self.nlp"
        actual_pos = doc_actual[-1].pos_  
        
        # This is where we initialize a baseline for the target word with no prior context
        if not candidate_predictions:
            return {
                # Retains the current word
                "actual_pos": actual_pos,
                # Sets a default probability
                "pos_prediction": 1.0,
                # Sets surprisal and entropy as 0 since we are on the 1st word
                "pos_surprisal": 0.0,
                "pos_entropy": 0.0,
                # Here we assing every POS category to its predicted likelihood under the language model's top prediction 
                # Since we have no prior context, we are assigning 100% (1.0) of the probability mass to our current word
                "pos_distribution": {actual_pos: 1.0}
            }

        # 2. Here our model put every candidate word into a sentence so that it can tag their syntactic category
        # it literally tries every word it has within a context 
        # With candidate words, it turns the dictionary into a standard python list
        candidate_words = list(candidate_predictions.keys())
        # Here it puts each word in its list into the prior context
        candidate_texts = [f"{context_text} {cand}".strip() for cand in candidate_words]
        
        # Here we are batching all of our cnadidate texts
        # Basically but every text into a matrix to process them simultaneously 
        docs = list(self.nlp.pipe(candidate_texts, batch_size=64))
        
        # We are setting up an empty dictionary 
        pos_distribution = {}
        # we are pulling out each candidate text with its candidate words
        for cand_w, doc in zip(candidate_words, docs):
            # We are selecting the last word (candidate word) and extract its syntactic POS tag 
            cand_pos = doc[-1].pos_
            # We are extracting its probability here 
            p = candidate_predictions[cand_w]
            # Here we are finalizing our loop. What we do here basically is this:
            # We take each candidate word, put them in context, extract their syntactic class, and by looking at their syntactic class, we are taking the sum of probability for every class. But summation is not carried out here 
            pos_distribution[cand_pos] = pos_distribution.get(cand_pos, 0.0) + p

        # Summation is carried out here and we start renormalization here.    
        total_mass = sum(pos_distribution.values())
        # Re-normalization step
        if total_mass > 0:
            # Divides each individual POS probability mass by total_mass 
            pos_distribution = {pos: mass / total_mass for pos, mass in pos_distribution.items()}
        else:
            # If total_mass is 0, it prevents a division by zero and assigns 100% of the probability
            pos_distribution = {actual_pos: 1.0}
        # Here we are getting the prediction values we calculated before
        # +1e-12: prevents taking log2(0)      
        pos_prediction = pos_distribution.get(actual_pos, 1e-12)
        # Here -np.log2()comptutes negative base-2 logarithm 
        pos_surprisal = -np.log2(pos_prediction)
        # We are calculating the entropy here 
        pos_entropy = -sum(p * np.log2(p) for p in pos_distribution.values() if p > 0)
        
        return {
            "actual_pos": actual_pos,
            "pos_prediction": pos_prediction,
            "pos_surprisal": pos_surprisal,
            "pos_entropy": pos_entropy,
            "pos_distribution": pos_distribution
        }


## --- Here is the 2nd part where we setting up our input to put into recipe ---


# We are following the same steps as before by setting device to cuda (software platform for accelaration used to program and run computations on GPU if available, otherwise use CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the pre-trained German GPT-2 model
model_name = "dbmdz/german-gpt2"

# 1. Initialize language model and POS predictor
tokenizer = AutoTokenizer.from_pretrained(model_name)
# AutoModelForCausalLM is a class for causal language modeling, which predicts the next token in a sequence based on the preceding context. The model is loaded onto the specified device (GPU or CPU) for efficient computation.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# Set the model to evaluation mode, which disables dropout (to ensure all neurons are active) and batch normalization (to ensure using fixed mean and variance).
# We use this to ensure the model's best  prediction performance as we are not training the model but using it.
model.eval()

# We are initiating our "device/recipe" which we defined before
pos_calculator = POSPredictor(spacy_model="de_dep_news_trf")

# Input german text
german_text = "Ich bin ein Berliner"
words = german_text.strip().split()

# Track word token spans for sliding context
# To calculate the surprisal of each word, we need to know which tokens correspond to that word in the tokenized sequence.
word_token_ids = []
for i, word in enumerate(words):
     # For each word, we encode it into token IDs using the tokenizer. We also add a space before the word if it's not the first word to ensure proper tokenization.
    # (i.e Word0: "Das" -> [token_id1, token_id2], Word1: "ist" -> [token_id3], Word2: "ein" -> [token_id4], Word3: "Beispiel." -> [token_id5, token_id6])
    prefix = " " if i > 0 else ""
    # tokenizer.encode: split word into tokens and maps to IDs
    token_ids = tokenizer.encode(prefix + word, add_special_tokens=False)
    word_token_ids.append(token_ids)

 # Flatten full token sequence by turning the nested list into the flat list([[464], [284, 15320, 814], [3488]] -> [464, 284, 15320, 814, 3488])
 # We don't lose the track because word_token_ids keeps track the word indices
all_tokens = [tok for sublist in word_token_ids for tok in sublist]

# 3. Word-by-word iteration for POS metrics
# Here we are defining our context length and overlap 
pos_results = []
current_token_idx = 0
max_context_len = 1024
min_overlap = 700

    # word_idx: an integer counter provided bu enumerate starting at 0, word: The actual word string, target_tokens: The list of integer subword token IDs 
for word_idx, (word, target_tokens) in enumerate(zip(words, word_token_ids)):
     # Determine how many tokens there are for each word
    target_len = len(target_tokens)

    # Determine context window boundaries (start at 0 or maintain >= min_overlap)
    # First row of this if loop is another way to say "if current_token_idx + target_len ≤ max_content_len"
    # Here what we mean by max_context_len is transformer's maximum for how many tokens they can process in a single forward pass (also called as initial context window)
    if current_token_idx <= max_context_len - target_len:
        # Executes when the target word i within that first window limit
        window_start = 0
    else:
      #Here is for the cases where we reach the maximum context length. This "else" case determines how many words to keep not to lose previous context and properly calulcate the surprisal.
        window_start = max(0, current_token_idx - min_overlap)
    # Store the preceding context   
    context_tokens = all_tokens[window_start:current_token_idx]
    # Store the preceding context + target wprd
    input_tokens = all_tokens[window_start:current_token_idx + target_len]
    context_text = tokenizer.decode(context_tokens).strip()

    # torch.tensor converts the standard Python integers into a Pytorch tensor so matrix multiplications and GPU operations can be performed on it
    # device=model.device matches the memory location of the input tensor to whereever the model parameters are stored ( gpu, cdu or coda)
    input_tensor = torch.tensor([input_tokens], device=device)

    # Disables PyTorch's autograd
    # Autograd is PyTorch's built-in engine that tracks operations performed on tensors to compute derivatives automatically 
    # This speed up the forward pass
    with torch.no_grad():
        # Executes the forward pass by returning a model output container which containts hidden representations, attention maps and raw prediction scores
        outputs = model(input_tensor)
        # Acceses the unnormalizd prediction scores (logits)
        # "outputs.logits" here has  a 3D tensor shape of (batch_size, sequence_length, vocab_size)
        # batch_size: the number of separate sentences processed in parallel 
        # sequence_length: The total count of subword tokens 
        # vocab_size: the total number of every possible next word in the dictionary 
        # i.e Input [1,256] -> Model Output Logits: [1,256,50257]
        # i.e batch_size: 1 (We are processing 1 sentence), sequence_length: 256 (The sentence is 256 tokens long), vocab_size: 50,257 ( For each position, the model outputs 50.257 numbers)
        # This code strips the first dimension yielding (sequence_length, vocab_size)
        logits = outputs.logits[0]
        
    candidate_predictions = {}
    # This if/else statement checks if there is any preceding text before the target word
    if len(context_tokens) > 0:
        # In Causal transformers, logits contains predictions for every position in a sentence
        # To get the prediction for k + 1, we have to know the prediction value at k
        next_token_logits = logits[len(context_tokens) - 1]
        # Applying softmax function to get raw  probabilities 
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Top-p (nucleus = 0.90) & Top-k (k >= 40) truncation[cite: 2]
        # We are sorting values in ascending or descending order 
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # We are taking the cumulative probability to normalize them later
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Try to find the first index where cumulative probability reaches or exceeds 0.90. .item(): converts the tensor returned into an integer and +1 is caried out since python uses 0-base indexing 
        cutoff_index = torch.searchsorted(cumulative_probs, 0.90).item() + 1
         # If you have small number of cutoff_index, model will still look the rest 40 - cutoff_index.
        # This is done to prevent turncation artifacts in enetropy or candidate logging
        # It is what we call Top-k truncation 
        k = max(40, cutoff_index)

        # Slices IDs of the first most likely k elements from the list
        top_indices = sorted_indices[:k].tolist()
        # Slices the first most likely k elements
        top_probs = sorted_probs[:k].tolist()
        # Total probability mass summation
        sum_probs = sum(top_probs)
        # We are performing renormalization again 
        norm_probs = [p / sum_probs for p in top_probs]

        # indexing and extracting normalized probs 
        for idx, p in zip(top_indices, norm_probs):
            # Converts indices to human readible names
            decoded = tokenizer.decode([idx]).strip()
             # We merge the probabilities when two different token IDs decode to the exact same word
            if decoded:
                candidate_predictions[decoded] = candidate_predictions.get(decoded, 0.0) + p



    # Compute POS metrics
    metrics = pos_calculator.compute_word_pos(
        context_text=context_text,
        actual_word=word,
        candidate_predictions=candidate_predictions
    )
    
    metrics["word_nmbr"] = word_idx + 1
    metrics["word"] = word
    pos_results.append(metrics)
    
    current_token_idx += target_len

# 4. Save to output.txt
output_file = "output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"{'Nmbr':<6}{'Word':<14}{'POS':<8}{'POS_Pred':<16}{'POS_Surp (bits)':<18}{'POS_Entropy (bits)':<20}\n")
    f.write("-" * 82 + "\n")
    for item in pos_results:
        f.write(
            f"{item['word_nmbr']:<6}"
            f"{item['word']:<14}"
            f"{item['actual_pos']:<8}"
            f"{item['pos_prediction']:<16.6e}"
            f"{item['pos_surprisal']:<18.4f}"
            f"{item['pos_entropy']:<20.4f}\n"
        )

print(f"Results successfully written to {output_file}")
