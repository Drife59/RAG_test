import json

from langchain_core.documents import Document

from src.models.mistral_models import MISTRAL_MEDIUM_31, frontier_mistral_client
from src.utils.utils import context_docs_to_dicts

prompt_validation_context = """
    Tu es un assistant juridique expert. Voici une requête utilisateur et une liste d'articles juridiques.
    Pour CHAQUE article, réponds avec un JSON contenant :
    - id_article (inchangé)
    - pertinent (booléen : vrai si l'article répond à la requête, faux sinon)
    - justification (une phrase expliquant pourquoi)

    **Requête utilisateur :**
    {query}

    **Liste des articles :**
    {articles}

    **Format de réponse attendu :**
    {{
      "evaluations": [
        {{
          "id_article": "ID1",
          "pertinent": true/false,
          "justification": "..."
        }},
        ...
      ]
    }}
"""


def get_prompt_batch(question: str, articles: list[Document]) -> str:
    dict_articles = context_docs_to_dicts(articles)
    json_articles = json.dumps(dict_articles, indent=2, ensure_ascii=False)

    prompt = prompt_validation_context.format(query=question, articles=json_articles)

    with open("prompt_reranking.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    return prompt


def evaluate_context(question: str, articles: list[Document]) -> list[dict]:
    prompt = get_prompt_batch(question, articles)
    response = frontier_mistral_client.chat.completions.create(
        model=MISTRAL_MEDIUM_31, messages=[{"role": "user", "content": prompt}]
    )
    response_content = response.choices[0].message.content

    # Can't evaluate context for some reason, just return articles
    if not response_content:
        # print("[reranking]/[evaluate_context]: could not evaluate context")
        raise Exception("[reranking]/[evaluate_context]: could not evaluate context")

    with open("result_reranking.txt", "w", encoding="utf-8") as f:
        f.write(response_content)

    # LLM tend to add extra marker at the beginning and end of file
    response_content = response_content.replace("```json", "")
    response_content = response_content.replace("```", "")

    return json.loads(response_content)
