from pathlib import Path
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


DOCUMENTS_PATH = Path.home() / "Documents"
PERSIST_DIR = "./chroma_db"

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
            loader = TextLoader(str(path), encoding="utf-8")

        return loader.load()

    except Exception as e:
        print(f"Failed: {path} -> {e}")
        return []


all_docs = []

files = [
    f for f in DOCUMENTS_PATH.rglob("*")
    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
]

print(f"Found {len(files)} files")

for file in tqdm(files):
    docs = load_file(file)

    for d in docs:
        d.metadata["source"] = str(file)

print("Done indexing")