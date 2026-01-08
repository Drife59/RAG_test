"""
Evaluate and filter context.

Use LLM to evaluate pertinence of context documents, regarding a question.
"""

import json

from langchain_core.documents import Document

from src.models.mistral_models import MISTRAL_MEDIUM_31, frontier_mistral_client
from src.utils.utils import context_docs_to_dicts

reranking_model = MISTRAL_MEDIUM_31
reranking_client = frontier_mistral_client

prompt_validation_context = """
    Tu es un assistant juridique expert. Voici une requête utilisateur et une liste d'articles juridiques.
    Pour CHAQUE article, réponds UNIQUEMENT avec un JSON contenant :
    - id_article (inchangé)
    - pertinent (booléen : vrai si l'article répond à la requête, faux sinon)
    - justification (une phrase expliquant pourquoi)

    J'insiste, renvoie UNIQUEMENT ce JSON.

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
    """Return a prompt the evaluate pertinence of a batch of context documents."""
    dict_articles = context_docs_to_dicts(articles)
    json_articles = json.dumps(dict_articles, indent=2, ensure_ascii=False)

    return prompt_validation_context.format(query=question, articles=json_articles)


def get_evaluated_context(question: str, articles: list[Document], write_raw_response: bool = False) -> list[dict]:
    prompt = get_prompt_batch(question, articles)
    response = reranking_client.chat.completions.create(
        model=reranking_model, messages=[{"role": "user", "content": prompt}]
    )
    response_content = response.choices[0].message.content

    # Maybe we shoud return all the context forced to pertinent
    if not response_content:
        raise Exception("[reranking]/[evaluate_context]: could not evaluate context")

    if write_raw_response:
        with open("result_reranking.txt", "w", encoding="utf-8") as f:
            f.write(response_content)

    # LLM tend to add extra marker at the beginning and end of file
    response_content = response_content.replace("```json", "")
    response_content = response_content.replace("```", "")

    return json.loads(response_content)["evaluations"]


def filter_context(evaluated_contexts: list[dict]) -> list[dict]:
    pertinent_count = 0

    non_pertinent_context_docs = []
    for evaluated_context in evaluated_contexts:
        if evaluated_context["pertinent"]:
            pertinent_count += 1
        else:
            non_pertinent_context_docs.append(evaluated_context)

    with open("non_pertinent_context.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(non_pertinent_context_docs, indent=2, ensure_ascii=False))

    print(f"pertinent context count: {pertinent_count}")
    print(f"filtered context count: {len(evaluated_contexts) - pertinent_count}")
    if len(evaluated_contexts) > 0:
        print(f"percentage pertinent context: {pertinent_count / len(evaluated_contexts)}")

    return [context for context in evaluated_contexts if context["pertinent"]]


def get_filtered_contexts(question: str, articles: list[Document]) -> list[Document]:
    evaluated_contexts = get_evaluated_context(question, articles)
    filtered_context = filter_context(evaluated_contexts)
    filtered_context_ids = [context["id_article"] for context in filtered_context]

    return [article for article in articles if article.metadata["article_id"] in filtered_context_ids]
