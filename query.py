from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


PERSIST_DIR = "./chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)


while True:
    query = input("\nAsk: ")

    if query.lower() in ["exit", "quit"]:
        break

    results = vectordb.similarity_search(query, k=5)

    print("\nTop Results:\n")

    for i, r in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Source: {r.metadata.get('source')}")
        print(r.page_content[:700])
        print("-" * 80)