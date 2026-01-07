import string
import random
import requests
from collections import defaultdict

class Ngram:
    def __init__(self, max_n=5):
        self.max_n = max_n
        
        # n -> context -> word -> count
        self.counts = {n: defaultdict(lambda: defaultdict(int)) for n in range(1, max_n + 1)}
        
        # n -> context -> total count
        self.context_counts = {n: defaultdict(int) for n in range(1, max_n + 1)}

    def process_line(self, line):
        line = line.lower().strip()
        line = line.translate(str.maketrans('', '', string.punctuation))
        return line.split()

    def train(self, text):
        text = text.replace('\n', ' ')
        tokens = self.process_line(text)

        for n in range(1, self.max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                context = ngram[:-1]   # empty tuple for unigram
                word = ngram[-1]
                if(len(context)==0):
                    continue
                self.counts[n][context][word] += 1
                self.context_counts[n][context] += 1

        print(f"Training complete! Learned n-grams from 1 to {self.max_n}")

    def predict(self, context):
        """
        context: tuple of words (can be length 0 to max_n-1)
        """
        context = tuple(context)
        max_context_len = min(len(context), self.max_n - 1)

        # Backoff: try highest order first
        for n in range(max_context_len + 1, 0, -1):
            ctx = context[-(n-1):] if n > 1 else ()
            
            if ctx in self.counts[n]:
                words = list(self.counts[n][ctx].keys())
                counts = list(self.counts[n][ctx].values())
                return random.choices(words, weights=counts, k=1)[0]

        return "<UNK>"

    def top_k_branching_contexts(self, n=5, k=3):
        return sorted(
            self.counts[n].items(),
            key=lambda item: len(item[1]),
            reverse=True
        )[:k]


# ------------------ DATA LOADING ------------------

url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = requests.get(url)
full_text = response.text

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK 1342 ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK 1342 ***"

if start_marker in full_text:
    full_text = full_text.split(start_marker)[1]
if end_marker in full_text:
    full_text = full_text.split(end_marker)[0]

# ------------------ TRAIN ------------------

ngram = Ngram(max_n=5)
ngram.train(full_text)

# ------------------ INTERACTIVE TEST ------------------

while True:
    line = input("Enter sentence (or press Enter): ").strip()
    if not line:
        break

    tokens = line.lower().split()
    context = tuple(tokens[-4:])  # max context for 5-gram
    print("Prediction:", ngram.predict(context))
