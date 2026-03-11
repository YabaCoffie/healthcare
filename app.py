import os
import streamlit as st
from rag import charger_csv, reponse



# --- EN-TÊTE DE L'APPLICATION ---
st.set_page_config(page_title="Assistant Médical MedQuAD", page_icon="🩺")

# Titre principal : Court et percutant
st.title("🩺 Assistant Médical Intelligent (RAG)")

# Sous-titre : Explique la technologie et la source
st.markdown("""
### Expertise basée sur le dataset certifié **MedQuAD** (NIH)
*Exploration de plus de 47 000 paires de questions-réponses médicales de confiance.*
""")

# Section des sources (utilisant ce que tu as trouvé sur Kaggle)
with st.expander("🔍 Voir les sources de données officielles"):
    st.write("""
    Cette IA synthétise des informations provenant de 12 sites des **National Institutes of Health (NIH)** :
    * **Cancer.gov** (Institut National du Cancer)
    * **CDC** (Centres pour le contrôle et la prévention des maladies)
    * **GARD** (Centre d'information sur les maladies génétiques et rares)
    * **NIDDK** (Diabète, maladies digestives et rénales)
    * **NHLBI** (Cœur, poumons et sang)
    * *Et d'autres sources majeures : NIHSeniorHealth, GHR, MPlusHealthTopics...*
    """)

# Une petite info-bulle pour la crédibilité
st.info("💡 L'IA utilise une chaîne d'orchestration pour harmoniser les réponses et garantir qu'elles proviennent de sources documentées.")

# --- RAPPEL DE SÉCURITÉ ---
st.warning("⚠️ **Avertissement :** Cet outil est un projet de recherche. Il ne remplace en aucun cas une consultation médicale.")
# text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = reponse(user_question)

    st.markdown("### Llama-3.3-70B Response")
    st.markdown(answer)
