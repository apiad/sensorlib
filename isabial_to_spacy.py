import spacy
import json
from spacy.tokens import DocBin


nlp = spacy.blank("en")


def convert_data(fname):
    print(f"Working on {fname}", flush=True)
    # the DocBin will store the example documents
    db = DocBin()

    with open(f"./Data/{fname}.jsonl") as fp:
        for i, line in enumerate(fp):
            print(f"\rLine {i}", end="...", flush=True)

            data = json.loads(line)
            text = data['text']
            annotations = data['tags']

            if not annotations:
                continue

            doc = nlp(text)
            ents = []

            for ann in annotations:
                span = doc.char_span(ann['start'], ann['end'], label=ann['tag'])
                ents.append(span)

                try:
                    doc.ents = ents
                except (ValueError, TypeError):
                    ents.pop()

            if doc.ents:
                db.add(doc)

    print("Saving...")
    db.to_disk(f"./Data/{fname}.spacy")


if __name__ == "__main__":
    convert_data("train")
    convert_data("dev")
    convert_data("test")
