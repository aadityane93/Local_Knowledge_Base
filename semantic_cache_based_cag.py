
from pathlib import Path
import ollama
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader,
)
from semantic_cache import SemanticCache

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_PATH = BASE_DIR / "PLACE_FILES_HERE"
MODEL_NAME = "gemma_4b_local:latest"

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".csv", ".py", ".js", ".ts", ".html", ".css", ".json",
}

def load_file(path):
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(path))
        elif suffix == ".csv":
            loader = CSVLoader(str(path))
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        return loader.load()
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return []

print("Source folder:", DOCUMENTS_PATH)
print("Loading documents...")
all_text = []
files = [
    file for file in DOCUMENTS_PATH.rglob("*")
    if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
]
for file in files:
    print("Loading:", file)
    docs = load_file(file)
    for doc in docs:
        all_text.append(f"Source: {file}\n{doc.page_content}")

context = "\n\n".join(all_text)
print(f"\nFiles loaded: {len(files)}  |  Context chars: {len(context)}\n")

# initialize the cache
cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=3600)

print("Cached CAG ready. Type 'exit' to quit, '!cache stats' for stats, '!cache clear' to clear.\n")

while True:
    question = input("Ask: ").strip()
    if question.lower() in ["exit", "quit"]:
        break
    if not question:
        continue
    if question == "!cache stats":
        print(cache.get_stats())
        continue
    if question == "!cache clear":
        cache.clear()
        print("Cache cleared.")
        continue

    # Check cache
    cached = cache.search(question)
    if cached:
        print("\n>>> (Cache Hit!) Answer:\n")
        print(cached["answer"])
        continue

    # No cache  then call LLM
    prompt = f"""You are a local CAG assistant. Use only the context below to answer.
If the answer is not in the context, say: "I cannot find that in the provided documents."
Context: {context}
Question: {question}
Answer: """
    response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0})
    answer = response["response"]

    cache.add(question, answer)

    print("\nAnswer:\n")
    print(answer)
