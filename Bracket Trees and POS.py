# Import the necessary libraries
import stanza

# Download the German models for the neural pipeline
stanza.download('de', package='default', processors='tokenize,pos,constituency')
nlp = stanza.Pipeline('de', processors='tokenize,pos,constituency')

# You paste your target text
text = """
"""
# Parse your text
doc = nlp(text)

# Makes an external txt file open and show your parsed data 
with open("output.txt", "w", encoding="utf-8") as f:
    # Initiate your for loop with enumerate function where "i" is your index and
    #"sentence" is your parsed sentence
    for i, sentence in enumerate(doc.sentences):
        
       # Get the constituency parse tree
        constituency_tree = sentence.constituency
        
        # Convert the Stanza tree to an NLTK tree
        bracket_string = str(constituency_tree)
        
        # Writes your output in some external txt 
        f.write(f"Constituency Tree of the Sentence {i + 1}:\n")
        f.write(f"[{i + 1}] {bracket_string}\n\n")

    
