import os
import sys

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import DB_PATH

# We need to do this trick, since python until 3.14 has sqlite3 3.31
# but Chroma requires 3.35+
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma  # noqa: E402


def _print_vector_caracteristics(vectorstore: Chroma):
    """ Warning: this function suppose all vectors have the same dimension. """
    collection = vectorstore._collection
    count = collection.count()
    # mypy is complaining it is not indexable but it actually is
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0] # type: ignore
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    
def create_embeddings(chunks: list[Document], langchain_embeddings: Embeddings) -> Chroma:
    if os.path.exists(DB_PATH):
        Chroma(persist_directory=DB_PATH.as_posix(), embedding_function=langchain_embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=langchain_embeddings, persist_directory=DB_PATH.as_posix()
    )

    _print_vector_caracteristics(vectorstore)
    return vectorstore