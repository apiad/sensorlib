import json
import string
import re
import numpy as np
from pathlib import Path
from openai import OpenAI


def build_taxonomy(fp):
    """
    Parse the CATEGORIES file and build the taxonomy list.

    Args:
    - `fp`: A file-like object pointing at the CATEGORIES list.

    Returns: A dictionary with all the categories as keys and their
    respective parent in the taxonomy as value.

    Remarks:
    The expected format is a plain text file with one category
    per line, indented one space for each level in the hierarchy.
    """
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

        i = parse(lines, i + 1, level + 1, line.strip())
        i = parse(lines, i, level, parent)

        return i + 1

    parse(lines)
    return categories


def parse_examples(path: str):
    """
    Parses annotated examples (in BRAT format) and constructs
    a JSON representation suitable for k-shot prompting.

    Args:
    - `path`: The path to the root folder of the dataset, i.e.,
    the folder that contains all .txt and .ann files.

    Returns:
    A list of dictionaries with all the examples and the
    corresponding annotations in the format expected by the prompt.
    """
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


def embed(client: OpenAI, texts: list[str], model:str):
    """
    Computes embeddings for a list of texts.

    Arguments:
    - `client`: An instance of `OpenAI` client configured with the right URL and key.
    - `texts`: A list of texts to embed.
    - `model`: The identifier of the embedding model to use.

    Returns: A numpy array of embeddings, one per row, in the same
    order as the input texts.
    """
    return np.asarray(_embed(client, texts, model))


def _embed(client: OpenAI, texts: list[str], model:str):
    """
    Recursive implementation of embeddings to avoid token limits.
    """
    if len(texts) < 20:
        try:
            embeddings = [
                e.embedding
                for e in client.embeddings.create(
                    input=texts, model=model
                ).data
            ]
            return embeddings
        except Exception:
            if len(texts) < 4:
                raise

    n = len(texts) // 2
    left = _embed(client, texts[:n], model)
    right = _embed(client, texts[n:], model)

    return left + right


def get_k_shot(client: OpenAI, text: str, examples: dict, embedding_model: str, embeddings: np.array, k: int):
    """
    Compute the best k-shot examples to use in a prompt.

    Arguments:
    - `client`: An instance of `OpenAI` client configured with the right URL and key.
    - `text`: The input text to analyze.
    - `examples`: The list of all examples obtained from `parse_examples`.
    - `embedding_model`: The identifier of the model used for embeddings.
    - `embeddings`: The numpy array of embeddings obtained from `embed`.
    - `k`: The number of examples to extract, e.g., `k=5`.

    Returns: The `k` best examples.

    Remarks: This method words by computing the cosine similarity between
    the input text and the examples. Beware that if you use one of the examples
    as input, it will probably come in the result.
    """
    x = embed(client, [text], embedding_model).T
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
    """
    Formats the prompt for NER using the input text, k-shot examples, and list of categories.

    Arguments:
    - `text`: The input text to analyze.
    - `categories`: The taxonomy as obtained from `build_taxonomy`.
    - `examples`: The k-shot examples obtained from `get_k_shot`.
    - `trim_categories`: If true, only the categories mentioned in the
    k-shot examples will be included in the prompt. This has the effect
    of reducing the prompt size significantly, which may improve recall
    in smaller language models.

    Returns: A string with the formatted prompt ready to
    be passed to the `reply` method.
    """
    base_classes = categories.keys()

    if trim_categories:
        base_classes = set([ann for e in examples for ann in e["annotations"].values()])

    base_classes = sorted(base_classes)

    examples = [
        EXAMPLE_PROMPT.format(
            text=k["text"], annotations=json.dumps(k["annotations"], indent=2)
        )
        for k in examples
    ]
    prompt = PROMPT.format(
        text=text,
        terms="\n".join(f"- {term}" for term in base_classes),
        examples="\n".join(examples),
    )
    return prompt


def reply(client: OpenAI, prompt: str, model: str):
    """
    Call the LLM and get a JSON response.

    Args:
    - `client`: An instance of `OpenAI` client configured with the right URL and key.
    - `prompt`: The actual prompt (see `build_prompt`).
    - `model`: The identifier of the LLM model to use.

    Returns: The parsed JSON response.
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages=messages, model=model, response_format=dict(type="json_object")
    )
    return json.loads(response.choices[0].message.content)


def convert_to_ann(text, annotations, categories):
    """
    Converts a JSON response to BRAT annotations.

    Arguments:
    - `text`: The input text.
    - `annotations`: The extracted annotations, as obtained from `reply`.
    - `categories`: The original taxonomy, as obtained from `build_taxonomy`.

    Returns: A BRAT-compatible text suitable for storing in .ann.
    """
    result = []

    for entity, label in annotations.items():
        if label not in categories:
            continue

        idx = len(result)
        parent = categories[label]
        parent2 = categories.get(parent, "")

        for match in re.finditer(entity, text):
            start, end = match.span()

            if (f" {text}"[start] in string.ascii_letters) or (
                f"{text} "[end] in string.ascii_letters
            ):
                continue

            result.append(f"T{idx}\t{label} {start} {end}\t{entity}")
            result.append(
                f"#{idx}\tAnnotatorNotes T{idx}\tCategoría: {parent}({parent2})"
            )

    return "\n".join(result)
