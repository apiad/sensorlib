from io import StringIO
from dotenv import load_dotenv

import streamlit as st
import sensorlib
from openai import OpenAI

load_dotenv()


@st.cache_resource
def build_llm():
    return OpenAI(base_url=st.secrets.openai.url, api_key=st.secrets.openai.key)


llm = build_llm()


categories_fp = st.sidebar.file_uploader("Categories file", "txt")

if categories_fp and st.sidebar.button("Build categories"):
    fp = StringIO(categories_fp.getvalue().decode("utf8"))
    st.session_state.categories = sensorlib.build_taxonomy(fp)

if "categories" not in st.session_state:
    st.error("Build the categories list first")
    st.stop()


categories = st.session_state.categories

with st.expander("Categories"):
    st.json(categories)


examples_path = st.text_input(
    "Training corpus path", "data/TECNOLOGIA/Anotaciones_v5_4_ITI"
)


@st.cache_data(show_spinner="Loading training set...")
def parse_examples(examples_path):
    return sensorlib.parse_examples(examples_path)


examples = parse_examples(examples_path)


@st.cache_data(show_spinner="Computing embeddings...")
def compute_embeddings(examples):
    return sensorlib.embed(
        llm, [e["text"] for e in examples], st.secrets.openai.embedding_model
    )


embeddings = compute_embeddings(examples)

with st.expander("Training corpus"):
    example_id = st.number_input("Example ID", min_value=0, max_value=len(examples) - 1)
    st.write(examples[example_id])
    st.write(embeddings[example_id])


input_text = st.text_area("Enter input text", examples[example_id]["text"], height=150)
k_shot = st.sidebar.number_input("K-shot", min_value=0, value=5)


@st.cache_data(show_spinner="Computing k-shot examples...")
def get_k_shot(text, k):
    k_shot_examples = sensorlib.get_k_shot(
        llm, input_text, examples, st.secrets.openai.embedding_model, embeddings, k=k + 1
    )
    return [e for e in k_shot_examples if e["text"] != text]


k_shot_examples = get_k_shot(input_text, k_shot)

with st.expander("K-shot examples"):
    st.write(k_shot_examples)


prompt = sensorlib.build_prompt(
    input_text,
    categories,
    k_shot_examples,
    trim_categories=st.sidebar.checkbox("Trim categories"),
)

with st.expander("Full prompt"):
    st.code(prompt)


@st.cache_data(show_spinner="Calling LLM..")
def call_llm(prompt):
    return sensorlib.reply(llm, prompt, model=st.secrets.openai.llm_model)


result = call_llm(prompt)
st.code(sensorlib.convert_to_ann(input_text, result, categories))
st.json(result)
