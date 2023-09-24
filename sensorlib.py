import pandas as pd
import jellyfish
import difflib

from dataclasses import dataclass


def taxonomy(path, nlp):
    data = pd.read_excel(path)
    return [Term(row[4], nlp, row[1], row[2], row[3]) for row in data.itertuples()]


class Term:
    def __init__(self, term, nlp, *categories) -> None:
        self.term = term
        self.doc = nlp(term)
        self.categories = set(categories) - {self.term}

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    def text(self):
        return self.term.lower()

    def __str__(self):
        if self.categories:
            return "%s (%s)" % (self.term, ", ".join(self.categories))

        return self.term


@dataclass
class Annotation:
    score: float
    term: Term
    surface: str
    start: int
    end: int

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Annotation) and (self.start, self.end) == (__o.start, __o.end)

    def to_dict(self):
        return dict(surface=self.surface, term=str(self.term), start=self.start, end=self.end, score=self.score)


def similarity(root:Term, term:Term, return_all=False):
    spacy_sim = root.similarity(term.doc)
    syntax_sim = jellyfish.jaro_winkler_similarity(root.text.lower(), term.text)
    diff_sim = difflib.SequenceMatcher(None, root.text.lower(), term.text).ratio()
    substr = 1 if term.text in root.text.split() else 0

    result = spacy_sim, syntax_sim, diff_sim, substr

    if return_all :
        return result

    return Annotation(score=max(result), term=term, surface=root.text, start=root[0].idx, end=root[0].idx+len(root.text))


def top_k_similar(root, set, k=10) -> list[Annotation]:
    sims = [similarity(root, term) for term in set]
    return sorted(sims, key=lambda t:t.score, reverse=True)[:k]


def top_k_terms(sentence, taxonomy, k=10) -> list[Annotation]:
    top_n = {}

    for e in sentence.noun_chunks:
        for ann in top_k_similar(e, taxonomy, k):
            if ann not in top_n or top_n[ann] < ann.score:
                top_n[ann] = ann.score

    return sorted(top_n, key=top_n.get, reverse=True)[:k]


def select_terms(terms, scores, threshold=0.5):
    mapping = { str(term.term): term for term in terms }
    result = {}
    used = set()

    for term, score in zip(scores['labels'], scores['scores']):
        if score < threshold:
            continue

        prefix, *_ = term.split("(")

        if prefix not in used:
            result[term] = score
            used.add(prefix)

    return { mapping[k]: v for k,v in result.items() }


def convert_to_brat(annotations):
    lines = []

    for i, annotation in enumerate(annotations):
        lines.append(f"T{i}\tKEYWORD\t{annotation.start} {annotation.end}\t{annotation.surface}")
        lines.append(f"#{i}\tAnnotatorNotes T{i}\t{str(annotation.term)}")

    return "\n".join(lines)


def spacy_to_brat(doc):
    lines = []

    for i, annotation in enumerate(doc.ents):
        lines.append(f"T{i}\t{annotation.label_}\t{annotation.start} {annotation.end}\t{annotation.text}")

    return "\n".join(lines)
