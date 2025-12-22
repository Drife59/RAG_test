import json
from pathlib import Path
from src.models.mistral_models import frontier_mistral_client
from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_3B
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

admin_message = """
    Tu es un assistant pour extraire des articles de documents juridiques.
    Tu reçois les documents juridiques en texte, et tu es capable
    d'identifier les articles et de les extraires.
"""

user_message = """
    Tu vas recevoir en context un morceau du code du travail français, un document juridique séparé en articles.
    Ta mission est de séparer chaque article de ce document.

    Dans ta réponse, donne moi le nombre d'article total que tu as trouvé.
    Puis donne moi, pour chaque article son identifiant et son contenu.

    Format de réponse obligatoire:
    Retourne UNIQUEMENT un JSON valide avec les clés suivantes :
    - "articles" : liste de dictionnaires, chaque dictionnaire contenant :
      - "id" : identifiant de l'article.
      - "content" : contenu textuel de l'article (chaîne de caractères).

    Voici des exemples d'identifiants d'articles:
    "Article L1111-2", "Article L1251-46", "Article D1221-25", "Article R1221-36", "Article D1226-8", "Article R4451-17"
    Contexte:
    {context}
"""


def get_prompt(file_path: Path) -> str:
    with open(file_path.as_posix(), 'r', encoding='utf-8') as f:
        file_content = f.read()

        return user_message.format(context=file_content)
    

def get_messages(file_path: Path) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": admin_message},
        {"role": "user", "content": get_prompt(file_path)},
    ]

def get_json_response(file_path: Path, model: str) -> dict | None:
    response_format: ResponseFormat = {"type": "json_object"}

    messages = get_messages(file_path)
    response = frontier_mistral_client.chat.completions.create(
        model=model, 
        messages=messages,
        response_format=response_format
    )
    response_content = response.choices[0].message.content
    
    if not response_content:
        return None
    
    return json.loads(response_content)

def get_articles(file_path: Path, model: str) -> list[dict]:
    json_response = get_json_response(file_path, model)

    if not json_response:
        return []

    # The last article is probably troncated, do not take it
    return json_response["articles"][:-1]

def index_article_by_id(articles: list[dict]) -> dict[str, str]:
    return {article["id"]: article["content"] for article in articles}


if __name__ == "__main__":
    test_path_164l = TXT_DIR / "code_du_travail_164l.txt"
    test_part1_path = TXT_DIR / "chunks/code_du_travail_part_1.txt"
    json_response = get_json_response(test_part1_path, MINISTRAL_3B)

    with open("result.txt", 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_response, indent=4, ensure_ascii=False))

    if not json_response:
        print("ERROR: LLM response empty, could not process file !")
        exit(1)
    articles = json_response["articles"][:-1]

    print(f"Nombre d'articles: {len(articles)}")