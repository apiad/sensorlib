import json
import string
import re
import numpy as np
from pathlib import Path
from openai import OpenAI
import streamlit as st

def build_taxonomy(fp):
    categories = {}
    lines = fp.readlines()

    def parse(lines, i=0, level=0, parent="ROOT"):
        if i >= len(lines):
            return i

        line = lines[i]
        my_level = line.count(" ")
        categories[line.strip()] = parent

        if my_level < level:
            return i

        i = parse(lines, i+1, level+1, line.strip())
        i = parse(lines, i, level, parent)

        return i+1

    parse(lines)
    return categories


def parse_examples(path):
    dataset = Path(path).glob("*.txt")
    examples = []

    for txt_file in dataset:
        ann_file = txt_file.with_suffix(".ann")

        with open(txt_file) as fp:
            text = fp.read().strip()

        with open(ann_file) as fp:
            annotations = {}

            for line in fp:
                if line.startswith("T"):
                    _, label, _, _, entity = line.strip().split(maxsplit=4)
                    annotations[entity] = label

        examples.append(dict(text=text, annotations=annotations))

    return examples


def reply(client, prompt, model=st.secrets.openai.llm_model):
    messages = [{"role":st.secrets.openai.role,"content":prompt}]
    response = client.chat.completions.create(
        messages=messages, model=model, response_format=dict(type="json_object")
    )
    return json.loads(response.choices[0].message.content)


def embed(client: OpenAI, texts: list[str]):
    return np.asarray(_embed(client, texts))


def _embed(client: OpenAI, texts: list[str]):
    if len(texts) < 20:
        try:
            embeddings = [e.embedding for e in client.embeddings.create(input=texts,model=st.secrets.openai.embedding_model).data]
            return embeddings
        except Exception:
            if len(texts) < 4:
                raise

    n = len(texts) // 2
    left = _embed(client, texts[:n])
    right = _embed(client, texts[n:])

    return left + right


def get_k_shot(client, text, examples, embeddings, k):
    x = embed(client, [text]).T
    scores = np.dot(embeddings, x).flatten()
    closest = np.argsort(scores)[-k:]
    return [examples[i] for i in closest]


PROMPT = """
The following is a list of categories from a taxonomy of terms referring to technology.
Your task is to extract relevant mentions of named entities from a text and classify them
according to this taxonomy.

# Terms

{terms}
- 99_OTHER

# Examples

{examples}

# Output format

Output your response in JSON format as shown in the example.

# Input

Text:
{text}

Answer:
"""

EXAMPLE_PROMPT = """
Text:
{text}

Answer:
{annotations}
"""


def build_prompt(text, categories, examples, trim_categories=False):
    base_classes = categories.keys()

    if trim_categories:
        base_classes = set([ann for e in examples for ann in e['annotations'].values()])

    base_classes = sorted(base_classes)

    examples = [EXAMPLE_PROMPT.format(text=k['text'], annotations=json.dumps(k['annotations'], indent=2)) for k in examples]
    prompt = PROMPT.format(text=text, terms="\n".join(f"- {term}" for term in base_classes), examples="\n".join(examples))
    return prompt


def convert_to_ann(text, annotations, categories):
    result = []

    for entity, label in annotations.items():
        if label not in categories:
            continue

        idx = len(result)
        parent = categories[label]
        parent2 = categories.get(parent, "")

        for match in re.finditer(entity, text):
            start, end = match.span()

            if (f" {text}"[start] in string.ascii_letters) or (f"{text} "[end] in string.ascii_letters):
                continue

            result.append(f"T{idx}\t{label} {start} {end}\t{entity}")
            result.append(f"#{idx}\tAnnotatorNotes T{idx}\tCategoría: {parent}({parent2})")

    return "\n".join(result)
