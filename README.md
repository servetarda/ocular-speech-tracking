## Pipeline for "Ocular Linguistic Tracking"

# Methodology
This GitHub repository is the part of a project where we look at the relationship between Ocular Speech Tracking and Linguistic units on multiple hierarchical levels. These levels consist of phonemes, syllables, words, syntax and semantics. This project's methodolgy is heavily - if not entirely - influenced by Heilbron et al. (2022) and it composed of 2 main parts: prediction calculation and mTRF analysis. Although mTRF analysis is quite common, prediction calculation follows the way Heilbron et al. showed. We are using predictor models calculating surprisal and entropy metrics for each level. Additionally, we are using both frequency and hierachy/previous context to calculate our metrics. Our environment is named nlp_env and we are using german_mfa pronounciation dictionary for lexical units in our story in German. 

# Files we touch 

1. run_story.py: Our actual machine! We give it a text file and it writes the finished CSVs.
   - pipeline.py: Ties the five levels together; splits our text into words
   - context_prior.py: Asks GPT-2 how likely every dictionary word is at each point in the story
   - lexical.py: The word-level probability machineru and the real-model backend
   - gpt2_onnx.py: A lighter alternate GPT-2 backend that skips PyTorch
   - phonemic.py: The sound-level competition model
   - sylabic.py: The syllable-level competition model
   - cohort.py: answers "given these candidates, how surprising is the next sound?"
   - syntactic.py: The grammar level
   - semantic.py: The meaning level (predicted vs actual word vector)
   - lexicon_mfa.py: Reads the MFA pronouciation dictionary which falls back to letter-to-sound for missing words
   - real_components_de.py: Wires up spaCy, wordfreq and GPT-2 specifically for German
   - mtrf_export.py: Explodes one-row-per-word into one-row-per-phoneme/per-syllable
   - textgrid_align.py: Merges real onset/offset times from a forced-alignment file
   - qa_export.py: Builds the human-review spot-check file.
   - config.py: Shared constants — bits vs. nats, truncation settings.
   - toy_resources.py · toy_resources_de.py: Fake miniature data used only by --backend toy.
   - __init__.py: Package bookkeeping.

3. story.txt: Our German story which is an actual input

4. result: Where our ouputs are written which is created the first time we run a full analysis. It includes words.csv, phonemes.csv, syllables.csv, qa_sample.csv, run_config.json
   - words.csv : lexical, pos, semantic + unigram_surprisal, contextual_dissimilairty
   - phonemes.csv : phon_surprisal_hier, phon_entropy_hier, phon_surprisal_freq,     phon_entropy_freq
   - syllables.csv : syll_surprisal_hier, syll_entropy_hier, syll_surprisal_freq, syll_entropy_freq
   - run-config.json : the receipt: exact settings used, vocab size, pronounciation coverage and the cohort-health check


# Environments we use 
nlp_env: Python, PyTorch, the German GPT-2 model, spaCY with German word vectors, word freqency data

# Runbook 

1. Go to you file and initiate your environment. In my case:

$ cd ~/linguistic_predictors
$ conda activate nlp_env
$ pwd
/Users/username/linguistic_predictors


2. If you wanna dry run first:

$ python run_story.py --text "Die kleine Katze rannte schnell unter den Baum." --backend toy --out results_toy/

3. Real model

$ head -20 story.txt > story_test.txt
$ python run_story.py --text-file story_test.txt --mfa-dict ~/Documents/MFA/pretrained_models/dictionary/german_mfa.dict --backend torch --gpt2-size small --device mps --batch-size 256 --no-g2p --vocab-size 10000 --out results_test/ --progress


