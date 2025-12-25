import os
from pathlib import Path

from langchain_core.documents import Document
from tqdm import tqdm  # type: ignore

from src.config import TXT_DIR
from src.embeddings.embed import create_embeddings
from src.models.mistral_models import MINISTRAL_3B as FRONTIER_MINISTRAL_3B
from src.models.mistral_models import frontier_mistral_client
from src.models.ollama_models import MINISTRAL3_3B, ollama_client
from src.preprocessing.cleaner.article_cleaner import clean_article_content
from src.preprocessing.extractor.article_extractor import SourcedArticle, get_sourced_articles, index_article_by_id
from src.rag_db.tables import Article

# From cleaning article, local model actually works fine
cleaner_client = ollama_client
cleaner_model = MINISTRAL3_3B 

extractor_client = frontier_mistral_client
extractor_model = FRONTIER_MINISTRAL_3B


def clean_articles(articles_by_id: dict[str, SourcedArticle]) -> dict[str, SourcedArticle]:
    print("Start cleaning articles...")

    cleaned_articles_by_id: dict[str, SourcedArticle] = {}
    for article_id, sourced_article in tqdm(articles_by_id.items()):
        cleaned_content = clean_article_content(cleaner_client, cleaner_model, sourced_article.content)
        cleaned_articles_by_id[article_id] = sourced_article
        cleaned_articles_by_id[article_id].content = cleaned_content

    print(f"{len(cleaned_articles_by_id)} articles cleaned.")
    return cleaned_articles_by_id

async def file_names_to_process(dir: Path) -> list[str]:
    """Check in db filename already processed and return list of file names to process."""
    file_names = os.listdir(dir.as_posix())

    db_articles = await Article.select(Article.id).output(as_list=True)
    file_names_processed = [article.source for article in db_articles]
    print("Following files has already been processed:", file_names_processed)

    filtered_file_names = [file_name for file_name in file_names if file_name not in file_names_processed]
    return [file_name for file_name in filtered_file_names]


def get_cleaned_articles_from_chunks(file_path: Path) -> dict[str, SourcedArticle]:
    print(f"Processing {file_path}...")
    sourced_articles = get_sourced_articles(file_path, extractor_client, FRONTIER_MINISTRAL_3B)
    print(f"{len(sourced_articles)} articles extracted from {file_path}.", flush=True)
    articles_by_id = index_article_by_id(sourced_articles)
    cleaned_articles_by_id = clean_articles(articles_by_id)

    return cleaned_articles_by_id

async def save_articles(articles_by_id: dict[str, SourcedArticle]) -> None:
    db_articles_ids = [article.id for article in await Article.objects()]
    print(f"There is already {len(db_articles_ids)} articles in the database.")
    articles_to_save = [article for article in articles_by_id.values() if article.id not in db_articles_ids]
    print(f"Saving {len(articles_to_save)} new articles.")
    articles_db = []
    for article in articles_to_save:
        new_article = Article(
            id=article.id,
            content=article.content,
            source=article.source
        )
        articles_db.append(new_article) 
    Article.insert(*articles_db)

async def get_articles_from_dir(dir: Path) -> dict[str, SourcedArticle]:
    # Avoid processing already processed files
    file_names = await file_names_to_process(dir)

    articles_by_id: dict[str, SourcedArticle] = {}
    for file_name in file_names:
        file_path = Path(dir.as_posix() + "/" + file_name)
        current_file_articles_by_id = get_cleaned_articles_from_chunks(file_path)
        await save_articles(current_file_articles_by_id)
        articles_by_id |= current_file_articles_by_id

    return articles_by_id

async def get_each_article_as_unique_doc(dir: Path) -> list[Document]:
    articles_by_id = await get_articles_from_dir(dir)

    documents = []
    for article_id, article in articles_by_id.items():
        documents.append(
            Document(
                page_content=article.content, 
                metadata={"article_id": article_id, "source": article.source}
            )
        )

    print(f'Found: {len(documents)} from processed files.')
    return documents


async def main():
    from src.embeddings.embedding_models import current_embedding_model
    chunk_dir = TXT_DIR / "chunks"

    documents = await get_each_article_as_unique_doc(chunk_dir)

    create_embeddings(documents, current_embedding_model)
    print("Ingestion complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())