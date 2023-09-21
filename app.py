import spacy
import streamlit as st
from streamlit.components.v1 import html

from spacy.tokens import DocBin
from spacy import displacy


st.set_page_config(page_title="Sensors", layout="wide", page_icon="🧠")


@st.cache_resource
def load_model():
    return spacy.load("Data/model-best")


@st.cache_data
def load_testing(_model):
    return list(DocBin().from_disk("Data/test.spacy").get_docs(_model.vocab))


with st.spinner("Loading model..."):
    nlp = load_model()


st.toast(f"Loaded spaCy model", icon="🌟")

with st.spinner("Loading data..."):
    testing = load_testing(nlp)


st.toast(f"Loaded {len(testing)} test sentences", icon="🌟")


example = st.slider(
    "Select example to visualize", min_value=0, max_value=len(testing) - 1
)
ground_truth = testing[example]

st.write("### Ground truth")
html(displacy.render(ground_truth, style="ent"))

predicted = nlp(ground_truth.text)

st.write("### Predicted")
html(displacy.render(predicted, style="ent"))
