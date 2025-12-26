import os
from pathlib import Path

from tqdm import tqdm  # type: ignore

from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_3B as FRONTIER_MINISTRAL_3B
from src.models.mistral_models import frontier_mistral_client
from src.preprocessing.extractor.article_extractor import SourcedArticle, get_sourced_articles, index_article_by_id
from src.rag_db.tables import Article

extractor_client = frontier_mistral_client
extractor_model = FRONTIER_MINISTRAL_3B


async def get_file_names_to_process(dir: Path) -> list[str]:
    """Check in db filename already processed and return list of file names to process."""
    file_names = os.listdir(dir.as_posix())

    file_names_processed = await Article.select(Article.source).distinct().output(as_list=True)
    print(f"Following files has already been processed: {file_names_processed} \n")

    filtered_file_names = [file_name for file_name in file_names if file_name not in file_names_processed]
    print(f"{len(filtered_file_names)} files will be processed.\n")
    return filtered_file_names

async def save_article(article: SourcedArticle) -> None:
    new_article = Article(
        id=article.id,
        content=article.content,
        source=article.source
    )
    try:
        # I prefer not to insert in bulk, to see any single article failure
        await Article.insert(new_article)
    except Exception as e:
        print("could not save article", new_article)
        print("Error:", e)

async def save_articles(articles_by_id: dict[str, SourcedArticle]) -> None:
    db_articles_ids = [article.id for article in await Article.objects()]
    print(f"There is already {len(db_articles_ids)} articles in the database.")
    articles_to_save = [article for article in articles_by_id.values() if article.id not in db_articles_ids]
    print(f"Saving {len(articles_to_save)} new articles.")

    for article in tqdm(articles_to_save):
        await save_article(article)

def get_articles_from_chunks(file_path: Path) -> dict[str, SourcedArticle]:
    print(f"\nProcessing {file_path}...")
    sourced_articles = get_sourced_articles(file_path, extractor_client, FRONTIER_MINISTRAL_3B)
    print(f"{len(sourced_articles)} articles extracted from {file_path}.", flush=True)
    article_by_id = index_article_by_id(sourced_articles)

    return article_by_id

async def fetch_articles_and_save_to_db(dir: Path) -> dict[str, SourcedArticle]:
    # Avoid processing already processed files
    file_names = await get_file_names_to_process(dir)

    article_by_id: dict[str, SourcedArticle] = {}
    for file_name in file_names:
        file_path = Path(dir.as_posix() + "/" + file_name)
        current_file_article_by_id = get_articles_from_chunks(file_path)
        await save_articles(current_file_article_by_id)
        article_by_id |= current_file_article_by_id

    return article_by_id


async def main():
    chunk_dir = TXT_DIR / "chunks"

    article_by_id = await fetch_articles_and_save_to_db(chunk_dir)

    print(f'{len(article_by_id)} articles saved to DB.')
    print("Ingestion from file to SQL complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())