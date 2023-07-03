import pandas as pd
import jellyfish
import difflib


def taxonomy(path, nlp):
    data = pd.read_excel(path)
    return [Term(row[4], nlp, row[1], row[2], row[3]) for row in data.itertuples()]


class Term:
    def __init__(self, term, nlp, *categories) -> None:
        self.term = term
        self.doc = nlp(term)
        self.categories = set(categories) - {self.term}

    @property
    def text(self):
        return self.term.lower()

    def __str__(self):
        if self.categories:
            return "%s (%s)" % (self.term, ", ".join(self.categories))

        return self.term


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


def top_k_terms(sentence, taxonomy, k=10):
    top_n = {}

    for e in sentence.noun_chunks:
        for term,w in top_k_similar(e, taxonomy, k):
            if term not in top_n or top_n[term] < w:
                top_n[term] = w

    return sorted(top_n, key=top_n.get, reverse=True)[:k]


def select_terms(scores, threshold=0.5):
    result = {}
    used = set()

    for term, score in zip(scores['labels'], scores['scores']):
        if score < threshold:
            continue

        prefix, *_ = term.split("(")

        if prefix not in used:
            result[term] = score
            used.add(prefix)

    return result
