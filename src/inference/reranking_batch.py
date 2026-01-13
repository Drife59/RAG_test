import json
from dataclasses import dataclass

from langchain_core.documents import Document

from src.models.mistral_models import MISTRAL_MEDIUM_31, frontier_mistral_client

reranking_model = MISTRAL_MEDIUM_31
reranking_client = frontier_mistral_client

METHOD = "POST"
reranking_url = str(reranking_client.base_url)

prompt_validation_context = """
    Tu es un assistant juridique expert. Voici une requête utilisateur et un article.
    Réponds UNIQUEMENT avec un JSON contenant :
    - id_article (inchangé)
    - pertinent (booléen : vrai si l'article répond à la requête, faux sinon)
    - justification (une phrase expliquant pourquoi)

    J'insiste, renvoie UNIQUEMENT ce JSON.

    **Requête utilisateur :**
    {query}

    **Article :**
    Article id: {article_id}
    Article content: {article_content}

    **Format de réponse attendu :**
    {{
        "id_article": "ID1",
        "pertinent": true/false,
        "justification": "..."
    }}
"""

example_json_line = {
    "custom_id": "request_1",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Explique-moi comment fonctionne un moteur électrique."}],
    },
}


@dataclass
class JsonLineBody:
    model: str
    messages: list[dict[str, str]]

    def to_dict(self) -> dict[str, str | list[dict[str, str]]]:
        return {
            "model": self.model,
            "messages": self.messages,
        }


@dataclass
class JsonLine:
    custom_id: str
    method: str
    url: str
    body: JsonLineBody

    def to_dict(self) -> dict:
        return {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def get_messages(article: Document, query: str) -> list[dict[str, str]]:
    prompt = prompt_validation_context.format(
        query=query, article_id=article.metadata["article_id"], article_content=article.page_content
    )
    return [{"role": "user", "content": prompt}]


def create_json_line_body(article: Document, query: str) -> JsonLineBody:
    messages = get_messages(article, query)
    return JsonLineBody(model=reranking_model, messages=messages)


def create_json_line(article: Document, query: str) -> JsonLine:
    return JsonLine(
        custom_id=article.metadata["article_id"],
        method=METHOD,
        url=reranking_url,
        body=create_json_line_body(article, query),
    )


if __name__ == "__main__":
    article_test = Document(
        page_content="Vous n'avez pas le droit de maltraiter les gens", metadata={"article_id": "test_id"}
    )
    test = create_json_line(article_test, "Le code du travail protège ?")

    print(test.to_json())
