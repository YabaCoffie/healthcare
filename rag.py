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
    vector_db = Chroma(
        embedding_function=embedding, 
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    retriever = vector_db.as_retriever()
    template = """
    Tu es un assistant IA médical expert. Ton rôle est de répondre aux questions de l'utilisateur 
    en utilisant exclusivement les extraits de la base de données MedQuAD fournis dans le contexte ci-dessous.

    Règles strictes :
    1. Si la réponse n'est pas présente dans le contexte, dis clairement que tu ne sais pas.
    2. N'utilise pas tes connaissances externes pour inventer des faits médicaux non présents ici.
    3. Reste professionnel, précis et synthétique.
    
    Contexte : {context}
    Question : {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(
        llm, prompt
    )  # va harmoniser les reponses
    retrieval_chain = create_retrieval_chain(
        retriever, document_chain
    )  # l'orchestreur final, va aller chercher la reponse à l'aide du pointeur retriever puis utilise l'harmoniseur
    reponse = retrieval_chain.invoke({"input": question})
    return reponse["answer"]
