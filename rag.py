import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
loader_csv = CSVLoader(
    "medquad.csv", source_column="source", metadata_columns=["focus_area"]
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)


def charger_csv():
    documents = loader_csv.load()
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore",
    )
    return 0


def reponse(question):
    prompt_traduction = f"Translate the following medical question to English. If it is already in English, return it as is. Return ONLY the translation, no preamble, no quotes: {question}"    
    traduction_result = llm.invoke(prompt_traduction)
    question_en = traduction_result.content.strip().replace('"', '').replace("'", "")
    # DEBUG : Affiche dans ta console ce que le moteur va chercher
    print(f"--- Recherche SQL/Vecteur pour : {question_en} ---")
    vector_db = Chroma(
        embedding_function=embedding, 
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    # TEMPLATE EN ANGLAIS POUR ÉVITER LES CONFUSIONS
    template = """
    You are an expert medical AI assistant. Your role is to answer the user's questions 
    using exclusively the MedQuAD database excerpts provided in the context below.

    Strict Rules:
    1. If the answer is not present in the context, clearly state that you do not know.
    2. Do not use external knowledge to invent medical facts not present here.
    3. Stay professional, precise, and concise.
    4. You must respond in English.
    
    Context: {context}
    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(
        llm, prompt
    )  # va harmoniser les reponses
    retrieval_chain = create_retrieval_chain(
        retriever, document_chain
    )  # l'orchestreur final, va aller chercher la reponse à l'aide du pointeur retriever puis utilise l'harmoniseur
    reponse = retrieval_chain.invoke({"input": question_en})
    return reponse["answer"]
