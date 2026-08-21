# Download and install the German model for Stanza if not already done
# Import the Stanza library
import stanza

# Initialize pipeline (download once beforehand if not cached)
nlp = stanza.Pipeline("de", processors="tokenize,pos,constituency")

text = """Hier steht der Beispieltext."""

# Process the text
doc = nlp(text)

# Write constituency parse trees to external output.txt
with open("output.txt", "w", encoding="utf-8") as f:
  # Loop through sentences and write constituency parse trees
  for i, sentence in enumerate(doc.sentences, start=1):
    # Get constituency parse tree in bracketed format
    bracket_string = str(sentence.constituency)

    # Get POS tags for each word in the sentence
    pos_tags = " ".join(
      f"{word.text}/{word.upos}" for word in sentence.words
    )

    # Write to file
    number = i + 1
    f.write(f"Sentence {number}:\n")
    f.write(f"[{number}] {bracket_string}\n\n")
