import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

# --- Configuration ---
DATA_PATH = "data"
VECTORSTORE_PATH = "faiss_store"

# --- Use Local Ollama Models ---
LLM = ChatOllama(model="llama3.1", temperature=0)
EMBED_MODEL = OllamaEmbeddings(model="nomic-embed-text")

RAG_PROMPT_TEMPLATE = """
You are an expert Paris bike tour guide. Use the following context from your curated guides to answer the user's question.
If the answer is not in the context, say "I don't have that specific information in my guides."

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

def get_vectorstore():
    """Creates or loads the FAISS vector store."""
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vector store from /faiss_store...")
        return FAISS.load_local(VECTORSTORE_PATH, EMBED_MODEL, allow_dangerous_deserialization=True)

    print("Creating new vector store...")
    loaders = []
    data_dir = Path(DATA_PATH)

    for path in data_dir.rglob("*.pdf"):
        loaders.append(PyPDFLoader(str(path)))
    for path in data_dir.rglob("*.txt"):
        loaders.append(TextLoader(str(path)))
    for path in data_dir.rglob("*.md"):
        loaders.append(TextLoader(str(path)))

    if not loaders:
        raise RuntimeError(f"No .txt, .pdf or .md files found in {DATA_PATH}")

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, EMBED_MODEL)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("Vector store created and saved to /faiss_store.")
    return vectorstore

def format_docs_with_sources(docs):
    """Format documents for the prompt with source information."""
    formatted = []
    sources = set()
    
    for doc in docs:
        # Extract filename from source path
        source = doc.metadata.get('source', 'unknown')
        filename = Path(source).name
        sources.add(filename)
        
        formatted.append(f"[Source: {filename}]\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(formatted)
    source_list = ", ".join(sorted(sources))
    
    # Return both context and sources
    return {
        "context": context,
        "sources": source_list
    }

def get_rag_chain():
    """Initializes and returns the RAG chain with source tracking."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Retrieve top 4 chunks

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Custom function to handle retrieval and format with sources
    def retrieve_and_format(query):
        docs = retriever.invoke(query)
        result = format_docs_with_sources(docs)
        return result["context"], result["sources"]
    
    # Modified RAG chain that returns both answer and sources
    def rag_with_sources(query):
        context, sources = retrieve_and_format(query)
        
        # Generate answer
        messages = prompt.format_messages(context=context, input=query)
        response = LLM.invoke(messages)
        answer = response.content
        
        # Append sources to answer
        if sources:
            answer += f"\n\n📚 Sources used: {sources}"
        
        return answer
    
    return rag_with_sources

# --- Main function to test this file directly ---
if __name__ == "__main__":
    # This code only runs when you execute `python rag_pipeline.py`
    print("\n--- RAG Pipeline Test (Local) ---")
    rag_chain = get_rag_chain()

    question = "What is a good scenic route for beginners?"
    response = rag_chain(question)
    print("\nQuestion:", question)
    print("Answer:", response)