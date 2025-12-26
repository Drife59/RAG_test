from pathlib import Path

from langchain_core.documents import Document

from src.config import TXT_DIR
from src.embeddings.embed import create_embeddings
from src.models.mistral_models import MINISTRAL_3B as FRONTIER_MINISTRAL_3B
from src.models.mistral_models import frontier_mistral_client
from src.rag_db.tables import Article

ingest_client = frontier_mistral_client
ingest_model = FRONTIER_MINISTRAL_3B


async def get_each_article_as_unique_doc(dir: Path) -> list[Document]:
    articles = await Article.objects()

    documents = []
    for article in articles:
        documents.append(
            Document(
                page_content=article.content, 
                metadata={"article_id": article.id, "source": article.source}
            )
        )

    return documents


async def main():
    from src.embeddings.embedding_models import current_embedding_model
    chunk_dir = TXT_DIR / "chunks"

    documents = await get_each_article_as_unique_doc(chunk_dir)

    print(f'Found: {len(documents)} from DB, to embed in vector DB.')

    create_embeddings(documents, current_embedding_model)
    print("Ingestion for SQL DB complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())