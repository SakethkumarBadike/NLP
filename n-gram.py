import string
import random
import requests

class Ngram:
    def __init__(self,n=3):
        self.n=n
        self.context_counts=dict()
        self.counts=dict()
    def process_line(self, line):
        line=line.lower().strip()
        line=line.translate(str.maketrans('','',string.punctuation))
        return line.split()
    def train(self,text):
            text = text.replace('\n', ' ') 
            tokens = self.process_line(text)
            for i in range(0, len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                word = ngram[-1]
                
                if context not in self.counts:
                    self.counts[context] = dict()
                if word not in self.counts[context]:
                    self.counts[context][word] = 0
                if context not in self.context_counts:
                    self.context_counts[context] = 0

                self.context_counts[context] += 1
                self.counts[context][word] += 1
        
            print(f"Training complete! Learned {len(self.context_counts)} unique contexts.")
    def predict(self,context):
             if(context not in self.counts):
                  return ["<UNK>"]
             counts=self.counts[context]
             possible_values=list(counts.keys())
             total=self.context_counts[context]
             prob=[count/total for count in counts.values()]
             return random.choices(possible_values,weights=prob,k=1)
    def top_k_branching_contexts(self, k=3):
        return sorted(
            self.counts.items(),
            key=lambda item: len(item[1]), 
            reverse=True
        )[:k]


        

url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = requests.get(url)
full_text = response.text
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK 1342 ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK 1342 ***"
if start_marker in full_text:
    full_text = full_text.split(start_marker)[1]
if end_marker in full_text:
    full_text = full_text.split(end_marker)[0]


ngram=Ngram(5)
ngram.train(full_text)

# print(ngram.top_k_branching_contexts())
line=input("enter sentence 1 : ")
print(ngram.predict(tuple(line.split(" ")[0:4])))
line=input("enter sentence 2 : ")
print(ngram.predict(tuple(line.split(" ")[0:4])))
line=input("enter sentence 3 : ")
print(ngram.predict(tuple(line.split(" ")[0:4])))
