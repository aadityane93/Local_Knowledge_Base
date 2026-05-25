from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from semantic_cache import SemanticCache

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
llm = Ollama(model="gemma_4b_local:latest")

# Initialize the cache
# for only 1hr caching
cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=3600)

print("Cached RAG ready. Type 'exit' to quit, '!cache stats' for stats, '!cache clear' to clear.\n")

# some startup message
while True:
    question = input("Ask: ")
    if question.lower() in ["exit", "quit"]:
        break
    if question.lower() == "!cache stats":
        print(cache.get_stats())
        continue
    if question.lower() == "!cache clear":
        cache.clear()
        print("Cache cleared.")
        continue

    # 1. Check semantic cache
    cached = cache.search(question)
    if cached:
        print("\n>>> (Cache Hit!) Answer:\n")
        print(cached["answer"])
        if cached.get("sources"):
            print("\nSources:")
            for s in cached["sources"]:
                print("-", s)
        continue

    # 2. Cache miss – do normal RAG retrieval + generation
    results = vectordb.similarity_search(question, k=5)
    context = "\n\n".join([
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in results
    ])
    prompt = f"""You are an assistant answering questions using only the provided context.
                Context: {context}
                Question: {question}
                Answer:
                """
    answer = llm.invoke(prompt)
    sources = [doc.metadata.get('source', 'Unknown') for doc in results]

    # 3. Store in cache
    cache.add(question, answer, sources)

    print("\nAnswer:\n")
    print(answer)
    print("\nSources:")
    for s in sources:
        print("-", s)