import os
import streamlit as st
from rag import charger_csv, reponse



# --- EN-TÊTE DE L'APPLICATION ---
st.set_page_config(page_title="MedQuAD Medical Assistant", page_icon="🩺")

# Titre principal
st.title("🩺 Intelligent Medical Assistant (RAG)")

st.markdown("""
### ### Expertise based on the certified **MedQuAD** (NIH) dataset
*Exploring over 47,000 trusted medical question-and-answer pairs.*
""")

# Section des sources
with st.expander("🔍 View official data sources"):
    st.write("""
    This AI synthesizes information from 12 **National Institutes of Health (NIH)** sites:
    * **Cancer.gov** (National Cancer Institute)
    * **CDC** (Centers for Disease Control and Prevention)
    * **GARD** (Genetic and Rare Diseases Information Center)
    * **NIDDK** (Diabetes, Digestive and Kidney Diseases)
    * **NHLBI** (Heart, Lung, and Blood Institute)
    * *And other major sources: NIHSeniorHealth, GHR, MPlusHealthTopics...*
    """)

# Une petite info-bulle pour la crédibilité
st.info("💡 The AI uses an orchestration chain to harmonize responses and ensure they come from documented sources.")

# --- RAPPEL DE SÉCURITÉ ---
st.warning("⚠️ **Warning:** This tool is a research project. It is not a substitute for a medical consultation.")
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = reponse(user_question)

    st.markdown("### Llama-3.3-70B Response")
    st.markdown(answer)
