from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings,
)

llm = Ollama(model="gemma_4b_local")


while True:
    question = input("\nAsk: ")

    if question.lower() in ["exit", "quit"]:
        break

    results = vectordb.similarity_search(question, k=5)

    context = "\n\n".join([
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in results
    ])

    prompt = f"""
You are an assistant answering questions using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")
    for doc in results:
        print("-", doc.metadata.get("source"))