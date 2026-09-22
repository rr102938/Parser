import nltk
import sys

nltk.download('punkt_tab')
nltk.download('punkt')

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> N | Det N | Adj NP | Det Adj NP | NP PP
VP -> V | V NP | V PP | V NP PP | VP Conj VP | V VP
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    s = preprocess(s)

    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    for tree in trees:
        tree.pretty_print()
        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    
    # Convert `sentence` to a list of its words.
    
    tokens = nltk.word_tokenize(sentence.lower())
    # Keep only words with at least one alphabetic character
    return [word for word in tokens if any(char.isalpha() for char in word)]


def np_chunk(tree):
    
    # Return a list of all noun phrase chunks in the sentence tree.
    
    chunks = []
    for subtree in tree.subtrees():
        # Identify subtrees labeled "NP"
        if subtree.label() == "NP":
            # A chunk is an NP that contains NO other NP as a subtree
            # We look at the subtree's own subtrees (excluding itself)
            if not any(child.label() == "NP" for child in subtree.subtrees() if child != subtree):
                chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
