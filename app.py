import streamlit as st
import pandas as pd
import spacy
from transformers import pipeline

# -----------------------------
# Load models only once
# -----------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    nlp = spacy.load("en_core_web_sm")
    return summarizer, nlp

summarizer, nlp = load_models()

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Medical NLP Project",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical NLP Project")
st.write("Medical Text Summarization and Entity Extraction")

text = st.text_area(
    "Enter Medical Report",
    height=250,
    placeholder="Paste medical text here..."
)

if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter some text.")
    else:

        # -----------------------------
        # Summarization
        # -----------------------------
        with st.spinner("Generating summary..."):
            summary = summarizer(
                text[:1000],
                max_length=60,
                min_length=20,
                do_sample=False
            )

        st.subheader("📄 Summary")
        st.success(summary[0]["summary_text"])

        # -----------------------------
        # SpaCy Entities
        # -----------------------------
        st.subheader("🏥 Named Entities")

        doc = nlp(text)

        entities = []

        for ent in doc.ents:
            entities.append({
                "Entity": ent.text,
                "Label": ent.label_
            })

        if entities:
            st.table(pd.DataFrame(entities))
        else:
            st.info("No entities found.")

        # -----------------------------
        # Rule-based Medical Detection
        # -----------------------------
        st.subheader("💊 Medical Keywords")

        diseases = [
            "diabetes",
            "allergy",
            "hypertension",
            "asthma"
        ]

        drugs = [
            "claritin",
            "zyrtec",
            "allegra",
            "insulin",
            "aspirin",
            "nasonex",
            "ortho tri-cyclen"
        ]

        symptoms = [
            "pain",
            "fever",
            "cough",
            "headache",
            "allergic"
        ]

        found = []

        lower = text.lower()

        for d in drugs:
            if d in lower:
                found.append([d, "Drug"])

        for d in diseases:
            if d in lower:
                found.append([d, "Disease"])

        for s in symptoms:
            if s in lower:
                found.append([s, "Symptom"])

        if found:
            st.table(
                pd.DataFrame(
                    found,
                    columns=["Keyword", "Type"]
                )
            )
        else:
            st.info("No medical keywords found.")