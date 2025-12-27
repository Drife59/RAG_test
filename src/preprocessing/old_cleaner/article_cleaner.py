"""
Allow the clean articles from unrellevant text, like chapter, title, section, etc.
"""

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from src.preprocessing.cleaner.content_article_cleaner import article_with_noise_both_side

from src.models.mistral_models import MINISTRAL_3B, frontier_mistral_client

admin_message = """
Tu es un assitant qui permet de "nettoyer" des articles juridiques.
Tu dois enlever les "chapitres" et les "parties", et garder uniquement le contenu de l'article.
Renvoie uniquement le contenu de l'article.
"""

user_message = """
Tu vas recevoir en context un article du code du travail.
Ta mission est de nettoyer cet article, en enlevant les chapitres et parties.
Ne change pas le contenu de l'article original.
Article:
{article}
"""

def get_messages(article_content: str) -> list[ChatCompletionMessageParam]:
    user_prompt = user_message.format(article=article_content)
    return [
        {"role": "system", "content": admin_message},
        {"role": "user", "content": user_prompt},
    ]

def get_response(client: OpenAI, model: str, messages: list[ChatCompletionMessageParam]) -> str | None:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def clean_article_content(client: OpenAI, model: str, article_content: str) -> str:
    cleaned_article_content = get_response(client, model, get_messages(article_content))

    if not cleaned_article_content:
        print("ERROR: LLM response empty, could not process article !")
        # Keep the original content, best effort
        return article_content

    return cleaned_article_content


if __name__ == "__main__":
    print(clean_article_content(frontier_mistral_client, MINISTRAL_3B, article_with_noise_both_side))