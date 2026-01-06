from langchain_core.documents import Document


def context_doc_to_dict(doc: Document) -> dict[str, str]:
    return {
        "article_id": doc.metadata["article_id"],
        "source": doc.metadata["source"],
        "page_content": doc.page_content,
    }


def dict_to_context_doc(doc: dict[str, str]) -> Document:
    return Document(
        page_content=doc["page_content"], metadata={"article_id": doc["article_id"], "source": doc["source"]}
    )


def context_docs_to_dicts(docs: list[Document]) -> list[dict[str, str]]:
    return [context_doc_to_dict(doc) for doc in docs]
