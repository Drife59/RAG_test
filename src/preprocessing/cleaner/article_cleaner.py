"""
Allow the clean articles from unrellevant text, like chapter, title, section, etc.
"""

from src.models.mistral_models import frontier_mistral_client
from src.models.mistral_models import MINISTRAL_3B
from openai import OpenAI
from src.preprocessing.cleaner.content_article_cleaner import article_with_noise_both_side

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

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

def get_response(client: OpenAI, messages: list[ChatCompletionMessageParam]) -> str | None:
    response = client.chat.completions.create(
        model=MINISTRAL_3B,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def clean_article(client: OpenAI, article: str) -> str | None:
    cleaned_article = get_response(client, get_messages(article))

    if not cleaned_article:
        print("ERROR: LLM response empty, could not process article !")
        return None

    return cleaned_article


if __name__ == "__main__":
    print(clean_article(frontier_mistral_client, article_with_noise_both_side))