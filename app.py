import streamlit as st
import json
from transformers import pipeline


st.set_page_config(page_title="Sensors", layout="wide", page_icon="🧠")


def load_jsonl(fname):
    with open(fname) as fp:
        return [json.loads(l) for l in fp]


@st.cache_data
def load_data():
    train = load_jsonl("Data/anotacionesFinalTrain.jsonl")
    eval = load_jsonl("Data/anotacionesFinalEval.jsonl")
    test = load_jsonl("Data/anotacionesFinalTest.jsonl")

    return train, eval, test


with st.spinner("Loading data..."):
    train, eval, test = load_data()


with st.expander("Training"):
    st.json(train)
