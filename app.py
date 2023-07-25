import streamlit as st
import spacy
from transformers import pipeline
from sensorlib import Annotation, taxonomy, top_k_terms, select_terms, convert_to_brat


st.set_page_config(page_title="Sensors", layout="wide", page_icon="🧠")


k_terms = st.sidebar.number_input("Candidate terms to extract", value=10)
threshold = st.sidebar.slider("Classification threshold", value=0.5, min_value=0.0, max_value=1.0)

@st.cache_resource
def load_parser():
    return spacy.load("en_core_web_lg")


@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@st.cache_data
def parse_document(_nlp, text):
    return _nlp(text)


@st.cache_resource
def load_taxonomy(_nlp):
    return taxonomy("ieee_taxonomy_v3.xlsx", _nlp)


text = st.text_area("Input document", height=10)

if not text:
    st.warning("Please introduce the text...")
    st.stop()

nlp = load_parser()
document = parse_document(nlp, text)
terms = load_taxonomy(nlp)
classifier = load_classifier()

sentences = list(document.sents)

cols = st.columns(4)
cols[0].metric("Sentences", len(sentences))
cols[1].metric("Terms in the Taxonomy", len(terms))


@st.cache_data
def extract_candidate_terms(sentence, count, _terms):
    return top_k_terms(sentence, _terms, count)


def score_terms(sentence, top_terms, _classifier):
    return _classifier(sentence, candidate_labels=[str(annotation.term) for annotation in top_terms], multi_label=True)


annotations = []


for sentence in sentences:
    st.info(sentence)

    with st.spinner("Extracting terms"):
        top_terms: list[Annotation] = top_k_terms(sentence, terms, k_terms)
        scores = score_terms(sentence.text, top_terms, classifier)

    for annotation, score in select_terms(top_terms, scores, threshold).items():
        st.write(annotation.term.text, score)
        annotation.score = score
        annotations.append(annotation)

    st.json(scores, expanded=False)

st.write("### Annotations")

st.table([annotation.to_dict() for annotation in annotations])

st.code(convert_to_brat(annotations))