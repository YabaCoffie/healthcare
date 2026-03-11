
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

## 🚀 Réalisations et Architecture RAG
L'approche initiale par mots-clés (TF-IDF) a été remplacée par un pipeline RAG (Retrieval-Augmented Generation) de pointe, capable de traiter des requêtes multilingues sur des sources anglophones :

* **Implémentation RAG & Vector Store :** Migration vers ChromaDB pour une recherche sémantique vectorielle. Le système n'est plus limité par les termes exacts mais comprend l'intention derrière chaque question.

* **Moteur de Recherche Translingue (Cross-Lingual) :** Utilisation d'embeddings multilingues (HuggingFace). Cette technologie permet à l'utilisateur de poser des questions en français tout en extrayant avec précision des informations pertinentes dans le dataset MedQuAD (majoritairement en anglais).

* **Intégration LLM & Synthèse :** Utilisation de Llama-3.3-70B (via Groq) comme cerveau décisionnel. Le modèle réalise une triple prouesse : il lit le contexte extrait, synthétise l'information médicale et retraduit dynamiquement la réponse dans la langue de l'utilisateur.

* **Garde-fous Médicaux :** Le pipeline est configuré avec des consignes de sécurité strictes pour interdire les hallucinations et garantir que chaque réponse est strictement ancrée dans les données officielles des NIH. Remplacer TF-IDF par des vecteurs d'embeddings contextuels (HuggingFace).

---

## 📦 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

---
