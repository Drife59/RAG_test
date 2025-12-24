import os
import time
from pathlib import Path

from langchain_core.documents import Document
from tqdm import tqdm  # type: ignore

from src.config import TXT_DIR
from src.embeddings.embed import create_embeddings
from src.models.mistral_models import MINISTRAL_3B as FRONTIER_MINISTRAL_3B
from src.models.mistral_models import frontier_mistral_client
from src.models.ollama_models import MINISTRAL3_3B, ollama_client
from src.preprocessing.cleaner.article_cleaner import clean_article
from src.preprocessing.extractor.article_extractor import get_articles, index_article_by_id

# From cleaning article, local model actually works fine
cleaner_client = ollama_client
cleaner_model = MINISTRAL3_3B 

extractor_client = frontier_mistral_client
extractor_model = FRONTIER_MINISTRAL_3B

# pause in sec after each chunk file article extraction
EXTRACTOR_PAUSE = 1

def clean_articles(articles_by_id: dict[str, str]) -> dict[str, str | None]:
    print("Start cleaning articles...")

    cleaned_articles_by_id: dict[str, str | None] = {}
    for article_id, content in tqdm(articles_by_id.items()):
        cleaned_articles_by_id[article_id] = clean_article(cleaner_client, MINISTRAL3_3B, content)

    print(f"{len(cleaned_articles_by_id)} articles cleaned.")
    return cleaned_articles_by_id

def get_articles_from_chunks(file_paths: list[str]) -> dict[str, str]:
    articles_by_id: dict[str, str] = {}
    for file_path in tqdm(file_paths):
        print(f"Processing {file_path}...")
        articles = get_articles(Path(file_path), extractor_client, FRONTIER_MINISTRAL_3B)
        articles_by_id |= index_article_by_id(articles)
        # trying to avoid 429 error
        time.sleep(EXTRACTOR_PAUSE)

    print(f"Il y a {len(articles_by_id)} articles récupéré du code du travail.")
    return articles_by_id

def get_each_article_as_unique_doc(dir: Path) -> list[Document]:
    file_names = os.listdir(dir.as_posix())
    # TODO: remove this when full process is possible
    file_paths = [dir.as_posix() + "/" + file_name for file_name in file_names][:1]
    print("[get_each_article_as_unique_doc] Number of files to process:", len(file_paths))

    article_by_id = get_articles_from_chunks(file_paths)
    cleaned_articles_by_id = clean_articles(article_by_id)

    documents = []
    for article_id, content in cleaned_articles_by_id.items():
        if content is None:
            continue
        documents.append(
            Document(page_content=content, metadata={"article_id": article_id})
        )

    print(f'Found: {len(documents)} from processed files.')
    return documents


if __name__ == "__main__":
    from src.embeddings.embedding_models import current_embedding_model
    chunk_dir = TXT_DIR / "chunks"

    documents = get_each_article_as_unique_doc(chunk_dir)

    create_embeddings(documents, current_embedding_model)
    print("Ingestion complete")