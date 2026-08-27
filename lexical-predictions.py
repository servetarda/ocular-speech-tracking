## 1. Contextual Lexical Predictions via Casual LLM ##

# This script computes the surprisal of a target word given a context using a causal language model (German GPT-2). 
# It also extracts the top-k/nucleus truncated candidate continuation distribution for the next word prediction.
# Top-k truncation limits the model to the k most probable next tokes 
# Top-p (nucleus) truncation limits the model to the smallest set of tokens whose cumulative probability exceeds a threshold p (e.g., 0.90).
# Unlike top-k, top-p truncation dynamically adjusts the number of candidates based on the model's uncertainty, allowing for more flexible and context-sensitive predictions.

# Import necessary libraries
# Deep learning framework by Meta for tensor computations and GPU acceleration
# Core tensor operations are computing dot products, matrix multiplications, applying activation functions, automatic differentiation and backpropagation.
import torch
# Provides a collection of functions for common operations on tensors, including activation functions, loss functions, and other utilities.
import torch.nn.functional as F
# Import the Hugging Face Transformers library for model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import NumPy for numerical operations
import numpy as np

# Set device to cuda (software platform for acceleration used to program and run computations on GPU) if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the pre-trained German GPT-2 model and tokenizer from Hugging Face's model hub. The model is designed for causal language modeling, which predicts the next token in a sequence based on the preceding context.
model_name = "dbmdz/german-gpt2"

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# AutoModelForCausalLM is a class for causal language modeling, which predicts the next token in a sequence based on the preceding context. The model is loaded onto the specified device (GPU or CPU) for efficient computation.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# Set the model to evaluation mode, which disables dropout (to ensure all neurons are active) and batch normalization (to ensure using fixed mean and variance).
# We use this to ensure the model's best  prediction performance as we are not training the model but using it.
model.eval()

german_text = "Ich bin ein Berliner"

def extract_full_text_lexical_predictions(text, tokenizer, model, max_context_len=1024, min_overlap=700):
    """
    Iterates through an entire text word-by-word, computing contextual surprisal
    and candidate distributions using a sliding context window.
    """
    # Tokenize the entire text while tracking word boundaries
    words = text.strip().split()
    results = []
    
    # Track word token spans which means the start and end token indices for each word in the tokenized sequence
    # To calculate the surprisal of each word, we need to know which tokens correspond to that word in the tokenized sequence.
    word_token_ids = []
    # For each word, we encode it into token IDs using the tokenizer. We also add a space before the word if it's not the first word to ensure proper tokenization.
    # (i.e Word0: "Das" -> [token_id1, token_id2], Word1: "ist" -> [token_id3], Word2: "ein" -> [token_id4], Word3: "Beispiel." -> [token_id5, token_id6])
    for i, word in enumerate(words):
        prefix = " " if i > 0 else ""
        token_ids = tokenizer.encode(prefix + word, add_special_tokens=False)
        word_token_ids.append(token_ids)
        
    # Flatten full token sequence by turning the nested list into the flat list ( [[464], [284, 15320, 814], [3488]] -> [464, 284, 15320, 814, 3488])
    # We don't lose the track because word_token_ids keeps track the word indices
    all_tokens = [tok for sublist in word_token_ids for tok in sublist]
    
    # Iterate word by word 
    current_token_idx = 0
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
        # Store the preceding context + target word
        input_tokens = all_tokens[window_start:current_token_idx + target_len]

        # torch.tensor converts the standard Python integers into a Pytorch tensor so matrix multiplications and GPU operations can be performed on it
        # device=model.device matches the memory location of the input tensor to whereever the model parameters are stored ( gpu, cdu or coda)
        input_tensor = torch.tensor([input_tokens], device=model.device)


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
            
        # This if/else statement checks if there is any preceding text before the target word
        if len(context_tokens) == 0:
            # Here with these two lines, we setting the default since there is no preceding context 
            log_prob_word = 0.0
            surprisal = 0.0
            entropy = 0.0
            # Extracts the prediction vectore produced by the first position by selecting the 1st token 
            next_token_logits = logits[0]
        else:
            ## Lecial Surprisal Calculation ##
            log_probs = []
            # The number  of preceding tokens before the current target word
            context_len = len(context_tokens)
            # This inner for loop calculates the exact log-probability for every subword token in the current target word using the chain rule of probabiltiy
            # target_tokens: the list of token IDs for the word
            # step: the subword counter 
            # target_tok: the specific subword token ID being evaluated at this step 
            for step, target_tok in enumerate(target_tokens):
                # the model uses the logits at the very last token of the preceding context to predict the first subword
                eval_pos = context_len - 1 + step
                # Extracts a 1D vector of size (vocab_size) containing the unnormalized scores for every possible word/subword at that position
                token_logits = logits[eval_pos]
                # Converts raw logits into log_probabilities (log P) across all vocabulary choices with SoftMax function
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                # Pulls out the log-probability corresponding to the true token ID by converting the 1-element PyTorch tensor into a standard Python float and collecting that score  inside log_probs
                log_probs.append(token_log_probs[target_tok].item())
            # Here we sum all log-probabilities of subword tokens making up the target word probability    
            log_prob_word = sum(log_probs)
            # We calculate the surprisal of the wholw word in bits
            # np.exp(log_prob_word): reverss the ln to recover the raw probability 
            # +1e-12: prevents taking log2(0) 
            #-np.log2(): comptutes negative base-2 logarithm 
            surprisal = -np.log2(np.exp(log_prob_word) + 1e-12)
            # Selects the model's raw output vector generated at the last token of the preceding context.
            # This will be used to compute entropy
            next_token_logits = logits[context_len - 1]
            # Converts the raw logits into a probability distribuiton 
            full_vocab_probs = F.softmax(next_token_logits, dim=-1)
            # Computes the base-2 logarithm of every toke's probability 
            full_vocab_log2_probs = torch.log2(full_vocab_probs + 1e-12)
            # -torch.sum(): sums the probability and log2 probability for all tokens 
            # .item(): converts the single-element PyTorch scalar tensor into a native Python float
            entropy = -torch.sum(full_vocab_probs * full_vocab_log2_probs).item()

        # Normalizes the unnormalized logits into probabilities    
        probs = F.softmax(next_token_logits, dim=-1)
        # Sorts all vocabulary probabilites from highest to lowest 
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # He are calculating the cumulative sum of the sorted possibilites to do Nucleus/Top-p Truncation
        # We are taking cumulative sum to see how many top words we need to reach a specific fraction
        # For example if 4 words account for 0.93 of total probability, we say that 4 words make up the majority of everything model thinks could come next
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Finds the index of first token where the cumulative probability reaches or exceeds the %90 nucleus threshold
        cutoff_index = torch.searchsorted(cumulative_probs, 0.90).item() + 1
        # If you have small number of cutoff_index, model will still look the rest 40 - cutoff_index.
        # This is done to prevent turncation artifacts in enetropy or candidate logging
        # It is what we call Top-k truncation 
        k = max(40, cutoff_index)

        # Slices IDs of the first most likely k elements from the list
        top_indices = sorted_indices[:k].tolist()
        # Slices the first most likely k elements
        top_probs = sorted_probs[:k].tolist()
        # Computes the total probability mass computed by these top_probs
        sum_probs = sum(top_probs)
        # Here we perform probability renormalization 
        # Since we multiply k words, their sum_probs is less than 1. To make their sum_probs equal to 1, we apply this renormalization.
        norm_probs = [p / sum_probs for p in top_probs]
        
        candidate_predictions = {}
        # We are initializing a for loop to assign indices and renormalized probability to idx and p
        for idx, p in zip(top_indices, norm_probs):
            # Converts indices to human readible text and use strip() to strip blank spaces
            decoded = tokenizer.decode([idx]).strip()
            if decoded:
                # We merge the probabilities when two different token IDs decode to the exact same word
                candidate_predictions[decoded] = candidate_predictions.get(decoded, 0.0) + p
        # We are assigning clean strings to objects        
        results.append({
            "word_number": word_idx + 1,
            "word": word,
            "surprisal": surprisal,
            "entropy": entropy,
            "probability": np.exp(log_prob_word),
            "candidates": candidate_predictions
        })
        # Shifts the token cursor forward by the number of subword tokeens in the current word  by ensuring the next word's evaluation starts at the correct position
        current_token_idx += target_len
        
    return results


# We are using our previously defined function
lexical_results = extract_full_text_lexical_predictions(
    text=german_text,
    tokenizer=tokenizer,
    model=model
)

output_file = "output.txt"


# Here we are drawing a clean table where we can see surprisal, entropy and probability
with open(output_file, "w", encoding="utf-8") as f:
    # 
    f.write(f"{'Nmbr':<6} {'Word':<18} {'Surprisal (bits)':<20} {'Entropy':<18} {'Probability':<15}\n")
    f.write("-" * 80 + "\n")
    
    # Write each word's metrics
    for item in lexical_results:
        f.write(
            f"{item['word_number']:<6} "
            f"{item['word']:<20} "
            f"{item['surprisal']:<18.4f} "
            f"{item['entropy']:<16.4f} "
            f"{item['probability']:<12.6e}\n"
        )

print(f"Results successfully written to {output_file}")
