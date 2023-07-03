#%%
import spacy
nlp = spacy.load('en_core_web_lg')

# %%
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# %%
sample = """Performance measurement systems have a critical role in organizations management, transforming data into relevant information for decision makers. In recent decades, the amount of data and information generated and shared has increased immensely, providing unprecedented opportunities and challenges for such systems. Faced with this scenario, this article aims to analyze the use of big data analytics in performance measurement systems to clarify the nexus between them. Furthermore, the aim is also to identify the trends and opportunities for future research. To achieve that, we carried a scientific map out using bibliometric analysis. The major results of the research show that the use of big data analytics in PMS has increased in recent years without considering the performance measurement systems characteristics. Incorporating artificial intelligence technologies such as machine learning and deep learning could improve the domain, creating opportunities for empirical works such as the use of unstructured data and applications in Industry 4.0."""
doc = nlp(sample)

#%%
import pandas as pd
taxonomy = pd.read_excel("ieee_taxonomy_v3.xlsx")

#%%
class Term:
    def __init__(self, term, *categories) -> None:
        self.term = term
        self.doc = nlp(term)
        self.categories = set(categories) - {self.term}

    @property
    def text(self):
        return self.term.lower()

    def __str__(self):
        if self.categories:
            return "%s in the context of %s" % (self.term, ", ".join(self.categories))

        return self.term

#%%
key_terms = [Term(row[4], row[1], row[2], row[3]) for row in taxonomy.itertuples()]

#%%
import jellyfish
import difflib

#%%
def similarity(root:Term, term:Term, return_all=False):
    spacy_sim = root.similarity(term.doc)
    syntax_sim = jellyfish.jaro_winkler_similarity(root.text.lower(), term.text)
    diff_sim = difflib.SequenceMatcher(None, root.text.lower(), term.text).ratio()
    substr = 1 if term.text in root.text.split() else 0

    result = spacy_sim, syntax_sim, diff_sim, substr

    if return_all :
        return result

    return max(result)

def top_k_similar(root, set, k=10):
    sims = {term:similarity(root, term) for term in set}
    return sorted(sims.items(), key=lambda t:t[1], reverse=True)[:k]

# %%
top_n = {}
sents = 0

for sent in doc.sents:
    print(sent)
    sents += 1

    for e in sent.noun_chunks:
        print(" - ", e)
        for term,w in top_k_similar(e, key_terms, 10):
            if term not in top_n or top_n[term] < w:
                top_n[term] = w

top = sorted(top_n, key=top_n.get, reverse=True)[:5 * sents]

for term in top:
    print(" * ", term)


# %%
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# %%
results = classifier(sample, candidate_labels=[str(term) for term in top], multi_label=True)

# %%
print(sample)

for term, score in zip(results['labels'], results['scores']):
    if score > 0.3:
        print(term, score)

# %%
