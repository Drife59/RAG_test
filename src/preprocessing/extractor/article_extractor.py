import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat

from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_3B, frontier_mistral_client

admin_message = """
    Tu es un assistant pour extraire des articles de documents juridiques.
    Tu reçois les documents juridiques en texte, et tu es capable
    d'identifier les articles et de les extraires.
"""

user_message = """
    Tu vas recevoir en context un morceau du code du travail français, un document juridique séparé en articles.
    Ta mission est de séparer chaque article de ce document.

    Donne moi, pour chaque article son identifiant et son contenu.

    Format de réponse obligatoire:
    Retourne UNIQUEMENT un JSON valide avec les clés suivantes :
    - "articles" : liste de dictionnaires, chaque dictionnaire contenant :
      - "id" : identifiant de l'article.
      - "content" : contenu textuel de l'article (chaîne de caractères).

    Voici des exemples d'identifiants d'articles:
    ["Article L1111-2", "Article L1251-46", "Article D1221-25", "Article R1221-36", "Article D1226-8",
      "Article R4451-17", "Article R5422-2-3"]
    Un identifiant d'article comporte obligatoirement le mot "Article".
    N'invente pas d'identifiant d'article si tu n'en trouve pas.
    
    Contexte:
    {context}
"""

@dataclass
class SourcedArticle:
    id: str
    content: str
    source: str # aka the chunk file it comes from

    @classmethod
    def from_dict(cls, data: dict[str, str], filename: str):
        return cls(
            id=data["id"],
            content=data["content"],
            source=filename
        )


def get_prompt(file_path: Path) -> str:
    with open(file_path.as_posix(), 'r', encoding='utf-8') as f:
        file_content = f.read()

        return user_message.format(context=file_content)
    

def get_messages(file_path: Path) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": admin_message},
        {"role": "user", "content": get_prompt(file_path)},
    ]

def get_json_response(file_path: Path, client: OpenAI, model: str, debug=False) -> dict | None:
    """
        Get the articles found from file, as:
        {
            "articles": [
                {
                    "id": "Article L1",
                    "content": "Tout projet (...)"
                },
                ...
            ]
        }
    """
    response_format: ResponseFormat = {"type": "json_object"}

    messages = get_messages(file_path)
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        response_format=response_format
    )
    response_content = response.choices[0].message.content
    
    if not response_content:
        return None
    
    if debug:
        with open("result.txt", 'w', encoding='utf-8') as f:
            f.write(json.dumps(response_content, indent=4, ensure_ascii=False))
    
    return json.loads(response_content)

def get_sourced_articles(file_path: Path, client: OpenAI, model: str) -> list[SourcedArticle]:
    json_response = get_json_response(file_path, client, model)

    if not json_response:
        return []
    
    # The last article is probably troncated, do not take it
    articles = json_response["articles"][:-1]
    sourced_articles = [SourcedArticle.from_dict(article, file_path.name) for article in articles]

    return sourced_articles

def index_article_by_id(articles: list[SourcedArticle]) -> dict[str, SourcedArticle]:
    return {article.id: article for article in articles}


if __name__ == "__main__":
    test_path_164l = TXT_DIR / "code_du_travail_164l.txt"
    test_part1_path = TXT_DIR / "chunks/code_du_travail_part_1.txt"
    sourced_articles = get_sourced_articles(test_part1_path, frontier_mistral_client, MINISTRAL_3B)

    for article in sourced_articles:
        print(f"ID: {article.id}")
        print(f"Content: {article.content}")
        print(f"Source: {article.source}")
        print("\n##########################\n")
    print(f"Nombre d'articles: {len(sourced_articles)}")