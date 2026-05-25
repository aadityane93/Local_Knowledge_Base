from pathlib import Path
import os

from tqdm import tqdm

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"

DOCUMENTS_PATH = BASE_DIR / "PLACE_FILES_HERE"

print("Source folder:", DOCUMENTS_PATH)
print("Chroma DB will be stored at:", PERSIST_DIR)


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".csv",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json",
}


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)


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
            loader = TextLoader(
                str(path),
                encoding="utf-8",
                autodetect_encoding=True,
            )

        return loader.load()

    except Exception as e:
        print(f"Failed: {path} -> {e}")
        return []


all_docs = []

files = [
    f for f in DOCUMENTS_PATH.rglob("*")
    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
]

print(f"\nFound {len(files)} supported files\n")


for idx, file in enumerate(tqdm(files, desc="Loading files"), start=1):
    print(f"\n[{idx}/{len(files)}] Processing file:")
    print(f"Folder: {file.parent}")
    print(f"File:   {file.name}")

    docs = load_file(file)

    print(f"Loaded document sections: {len(docs)}")

    for doc in docs:
        doc.metadata["source"] = str(file)
        doc.metadata["filename"] = file.name
        doc.metadata["folder"] = str(file.parent)

    all_docs.extend(docs)


print("\nFinished loading files")
print("Total loaded documents:", len(all_docs))


if len(all_docs) == 0:
    raise ValueError("No documents were loaded. Chroma DB was not created.")


print("\nSplitting documents into chunks...")

chunks = splitter.split_documents(all_docs)

print("Total chunks created:", len(chunks))


if len(chunks) == 0:
    raise ValueError("No chunks were created. Chroma DB was not created.")


os.makedirs(PERSIST_DIR, exist_ok=True)

print("\nCreating Chroma database...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(PERSIST_DIR),
)

try:
    vectordb.persist()
except Exception:
    pass


print("\nDone indexing")
print("Persist directory:", os.path.abspath(PERSIST_DIR))
print("Files inside persist directory:")

if os.path.exists(PERSIST_DIR):
    print(os.listdir(PERSIST_DIR))
else:
    print("Directory does not exist")
