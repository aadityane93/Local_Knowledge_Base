Place your industrial manuals, technical documentation, or reference materials in the `PLACE_FILES_HERE` directory.

`python -m venv venv`
`pip install -r requirements.txt`
`venv/Scripts/activate`



How it works:

1. Indexer.py    → prepares the RAG vector database

                  Documents folder
                     ↓
                  Find supported files
                     ↓
                  Load each file
                     ↓
                  Add metadata:
                       - source
                       - filename
                       - folder
                     ↓
                  Split documents into chunks
                     ↓
                  Create embedding for each chunk
                     ↓
                  Store chunks + embeddings in Chroma DB


2. Rag_based_query.py   → asks questions using retrieved chunks

                  User question
                     ↓
                  Create embedding for the question
                     ↓
                  Search Chroma DB
                     ↓
                  Find top 5 most related chunks
                     ↓
                  Join those chunks into context
                     ↓
                  Send context + question to Ollama LLM
                     ↓
                  LLM answers using only retrieved chunks
                     ↓
                  Print answer
                  Print sources


3. Cag_based_query.py  asks questions using the full loaded context
                
                  Documents
                     ↓
                  Load complete document text
                     ↓
                  Combine everything into one large context
                     ↓
                  Keep context in memory
                     ↓
                  User question
                     ↓
                  Send full context + question to LLM
                     ↓
                  LLM answers from the full context


4. Semantic_based_caching,py    →  Before doing RAG or CAG, first check whether a similar question was already answered.

                  Previous question
                     ↓
                  Question embedding
                     ↓
                  Answer
                     ↓
                  Optional source chunks
                     ↓
                  Timestamp
                     ↓
                  New user question
                     ↓
                  Create embedding for the question
                     ↓
                  Search previous cached question embeddings
                     ↓
                  Is a similar question already answered?
                     ↓
                  If yes, Return cached answer
                     ↓
                  If No, Run RAG or CAG
                     ↓
                  Generate answer
                     ↓
                  Save question + embedding + answer in cache
