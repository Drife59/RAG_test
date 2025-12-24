import os
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm  # type: ignore

from src.config import DB_PATH, TXT_DIR
from src.models.mistral_models import MINISTRAL_3B, frontier_mistral_client
from src.preprocessing.cleaner.article_cleaner import clean_article
from src.preprocessing.extractor.article_extractor import get_articles, index_article_by_id

# We need to do this trick, since python until 3.14 has sqlite3 3.31
# but Chroma requires 3.35+
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma  # noqa: E402


def fetch_documents_old(dir: Path) -> list[Document]:
    print(f"✓ Fetching documents from dir {dir}")
    documents = []

    loader = DirectoryLoader(
        path=dir.as_posix(), 
        glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    folder_docs = loader.load()

    for doc in folder_docs:
        documents.append(doc)
        filename = Path(doc.metadata["source"]).stem
        print(f'{filename} has been loaded')
    return documents

def clean_articles(articles_by_id: dict[str, str]) -> dict[str, str | None]:
    print("Start cleaning articles...")

    cleaned_articles_by_id: dict[str, str | None] = {}
    for article_id, content in tqdm(articles_by_id.items()):
        cleaned_articles_by_id[article_id] = clean_article(frontier_mistral_client, MINISTRAL_3B, content)

    print(f"{len(cleaned_articles_by_id)} articles cleaned.")
    return cleaned_articles_by_id

def get_articles_from_chunks(file_paths: list[str]) -> dict[str, str]:
    articles_by_id: dict[str, str] = {}
    for file_path in tqdm(file_paths):
        print(f"Processing {file_path}...")
        articles = get_articles(Path(file_path), frontier_mistral_client, MINISTRAL_3B)
        articles_by_id |= index_article_by_id(articles)

    print(f"Il y a {len(articles_by_id)} articles récupéré du code du travail.")
    return articles_by_id

def fetch_documents(dir: Path) -> list[Document]:
    # TODO: remove this :5
    file_names = os.listdir(dir.as_posix())[:5]
    file_paths = [dir.as_posix() + "/" + file_name for file_name in file_names]
    print("Number of files:", len(file_paths))

    article_by_id = get_articles_from_chunks(file_paths)
    cleaned_articles_by_id = clean_articles(article_by_id)

    print(f'Article count: {len(cleaned_articles_by_id)}')

    return []

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


if __name__ == "__main__":
    # from src.embeddings.embedding_models import current_embedding_model
    chunk_dir = TXT_DIR / "chunks"

    documents = fetch_documents(chunk_dir)

    # create_embeddings(chunks, current_embedding_model)
    print("Ingestion complete")