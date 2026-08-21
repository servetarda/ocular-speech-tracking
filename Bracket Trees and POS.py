# Import the necessary libraries
import string
from nltk.tree import Tree
import stanza

# Download the German models for the neural pipeline
stanza.download('de', package='default', processors='tokenize,pos,constituency')
nlp = stanza.Pipeline('de', processors='tokenize,pos,constituency')

text = """ 
"""

# Here are the punctuation tags used in the Stanza constituency parse trees
# We are defining them so that we can filter them out 
PUNCT_TAGS = {"$.", "$,", "$(", "$)", "$:", "$;", "$,", "$?", "$!", "$\"", "$'", "$`", "$``", "$''", "$--", "$-LRB-", "$-RRB-", "$-LSB-", "$-RSB-", "$-LCB-", "$-RCB-"}

def remove_punctuation(tree):
# Here, Tree is a list-like python object. Internal node or POS tag is an instance of Tree
# while terhminal nodes/leafs are strings 
  if not isinstance(tree, Tree):
    return tree

  filtered_children = []
  for child in tree:
    if isinstance(child, Tree):
      if child.label() in PUNCT_TAGS or (
        # We write this condition to confirm that the child is a pre-terminal node
        # pre-terminal puncuation nodes look like this: ($.)
          len(child) == 1
          # This is to check if the child has only one child, which is a string (the punctuation mark)
          and isinstance(child[0], str)
          # This is to check if the child is a punctuation mark
          and child[0] in string.punctuation
      ):
        # If the child is a punctuation node, we skip it and continue with non-punctuation children
        continue
      # If the child is not a punctuation node, we recursively call the function to check its children
      pruned_child = remove_punctuation(child)
      # If the pruned child is not None and has children, we add it to the filtered children list
      if pruned_child is not None and len(pruned_child) > 0:
        # We add the pruned child to the filtered children list
        filtered_children.append(pruned_child)
    else:
      # If the child is a string (terminal node), we check if it is a punctuation mark
      if child not in string.punctuation:
        # If the child is not a punctuation mark, we add it to the filtered children list
        filtered_children.append(child)
   # Finally, we return a new Tree with the same label as the original tree and the filtered children
  return Tree(tree.label(), filtered_children) if filtered_children else None

# Process the text with the Stanza pipeline
doc = nlp(text)

# Write the constituency trees to a file named "output.txt" in the current working directory
with open("output.txt", "w", encoding="utf-8") as f:
  # Loop through each sentence in the processed document
  for i, sentence in enumerate(doc.sentences):
    # Convert Stanza tree string representation into an NLTK Tree
    raw_nltk_tree = Tree.fromstring(str(sentence.constituency))

    # Prune punctuation nodes
    cleaned_tree = remove_punctuation(raw_nltk_tree)

    # Format the cleaned tree into a single-line bracketed string
    bracket_string = (
        " ".join(str(cleaned_tree).split()) if cleaned_tree else ""
    )

    # Write to file
    sentence_num = i + 160
    f.write(f"Constituency Tree of the Sentence {sentence_num}:\n")
    f.write(f"[{sentence_num}] {bracket_string}\n\n")
    
