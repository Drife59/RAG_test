import os
import sys
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings.embedding_models import (
    langchain_embedding_model_factory
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.embeddings import Embeddings
from src.config import DB_NAME, TXT_DIR
from pathlib import Path

# We need to do this trick, since python until 3.14 has sqlite3 3.31
# but Chroma requires 3.35+
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma # noqa: E402


def fetch_documents(dir: Path) -> list[Document]:
    print(f"✓ Fetching documents from dir {dir}")
    documents = []

    loader = DirectoryLoader(dir.as_posix(), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    folder_docs = loader.load()

    for doc in folder_docs:
        documents.append(doc)
        filename = Path(doc.metadata["source"]).stem
        print(f'{filename} has been loaded')
    return documents


def create_chunks(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    return chunks

def _print_vector_caracteristics(vectorstore: Chroma):
    """ Warning: this function suppose all vectors have the same dimension. """
    collection = vectorstore._collection
    count = collection.count()
    # mypy is complaining it is not indexable but it actually is
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0] # type: ignore
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    
def create_embeddings(chunks: list[Document], langchain_embeddings: Embeddings) -> Chroma:
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=langchain_embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=langchain_embeddings, persist_directory=DB_NAME
    )

    _print_vector_caracteristics(vectorstore)
    return vectorstore


if __name__ == "__main__":
    from src.embeddings.embedding_models import INTFLOAT_MULTILINGUAL_E5_LARGE
    embeddings = langchain_embedding_model_factory(INTFLOAT_MULTILINGUAL_E5_LARGE)
    documents = fetch_documents(TXT_DIR)
    chunks = create_chunks(documents)
    create_embeddings(chunks, embeddings)
    print("Ingestion complete")