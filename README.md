
---

# 🩺 MedQuAD Health Assistant: Analyse et QA par Recherche Sémantique

Ce projet utilise le dataset **MedQuAD** (Medical Question-Answering Dataset) pour construire un système de réponse aux questions médicales. L'approche actuelle repose sur une analyse exploratoire approfondie (EDA) et un moteur de recherche de réponses basé sur la similarité statistique (**TF-IDF**).

## 📊 Exploration et Analyse des Données (EDA)

Le projet commence par une analyse rigoureuse de **16 412 paires questions-réponses** médicales.

### 🔍 Statistiques Clés :

* **Sources :** 9 sources majeures des NIH (GHR, GARD, CancerGov, etc.).
* **Diversité :** 4 610 zones de focus médicales différentes.
* **Hétérogénéité :** Les réponses varient de **6 à 29 046 caractères**.

### 📈 Visualisations :

* **Distribution des longueurs :** Une forte asymétrie à droite (médiane à 882 caractères), révélant que quelques réponses sont très détaillées tandis que la majorité est concise.
* **Analyse par Source :** GARD et GHR dominent le dataset, tandis que CancerGov présente la plus grande variété de longueurs de texte.
* **Analyse Lexicale :** Identification des signatures sémantiques par source (ex: "cancer" pour CancerGov, "gene" pour GHR).

---

## 🛠️ Prétraitement & Pipeline de Données

Pour assurer la qualité des vecteurs, un nettoyage spécifique a été appliqué :

1. **Nettoyage :** Suppression des valeurs manquantes (NaN) et mise en minuscules.
2. **Normalisation :** Suppression de la ponctuation pour réduire le bruit.
3. **Split :** Division en ensembles d'entraînement (80%) et de test (20%).

---

## 🤖 Modélisation : Retrieval-Based QA

L'approche actuelle implémente un moteur de recherche de réponse "naïf" mais efficace pour établir une ligne de base (baseline).

### Méthodologie :

* **Vectorisation :** Utilisation de `TfidfVectorizer` avec des unigrammes et bigrammes ().
* **Pondération :** Application de la mise à l'échelle sous-linéaire () pour limiter l'impact des répétitions de mots.
* **Mesure :** Calcul de la **similarité cosinus** entre le vecteur de la question posée et l'ensemble de la matrice des réponses.

### Performance Actuelle :

* **Mean Cosine Similarity (Test) :** **0.21**
* **Observation :** Le modèle parvient à identifier des mots-clés (ex: "breast cancer"), mais manque de compréhension sémantique profonde. C'est un point de départ qui démontre la nécessité d'utiliser des modèles de langage plus avancés (LLM/RAG).

---

## 🚀 Prochaines Étapes

1. **Implémentation RAG :** Migration vers **LlamaIndex** et **ChromaDB** pour une recherche sémantique plus fine.
2. **LLM Integration :** Utilisation de modèles comme **Llama-3.3** ou **Gemini** pour reformuler et synthétiser les réponses extraites.
3. **Embeddings :** Remplacer TF-IDF par des vecteurs d'embeddings contextuels (HuggingFace).

---

## 📦 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

---
