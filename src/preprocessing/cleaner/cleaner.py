"""
Clean part file.
This is because extraction sometimes goes wrong, because of useless text.
This module try to remove it (titre, chapitre, section, livre).
"""

from pathlib import Path

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_14B, frontier_mistral_client

admin_message = """
Tu es un assistant qui permet de retirer les lignes de livre, titre, chapitre, section et sous section.
"""

user_message = """
Tu vas recevoir en contexte un texte comportant des articles juridiques.
Ton rôle est de retirer les lignes de livre, titre, chapitre, section et sous section.
Ne modifie pas le texte d'origine.
Renvoie le texte modifié sans autre commentaire.

Exemple de ligne de livre:
    - Livre II : La négociation collective - Les conventions et accords
collectifs de travail
    - Livre III : Les institutions représentatives du personnel
    - Livre V : Les conflits collectifs

Exemple de ligne de titre:
    - Titre Ier : Exercice du droit de grève
    - Titre III : Mesures de coordination avec les autres collectivités
ultramarines
    - Titre III : Repos et jours fériés

Exemple de ligne de chapitre:
    - Chapitre II : Principe de non-discrimination.
    - Chapitre Ier : Champ d'application.
    - Chapitre III : Plan et contrat pour l'égalité professionnelle

Exemple de ligne de section:
    - Section unique : Plan pour l'égalité professionnelle.
    - Section 1 : Dispositions générales.
    - Section 3 : Formalités à l'embauche et à l'emploi

Exemple de ligne de sous section:
    - Sous-section 2 : Négociation annuelle
    - Sous-section 3 : Négociation triennale
    - Sous-section 1 : Modalités de la négociation obligatoire

Contexte:
{Context}
"""

def get_messages(file_content: str) -> list[ChatCompletionMessageParam]:
    user_prompt = user_message.format(Context=file_content)
    return [
        {"role": "system", "content": admin_message},
        {"role": "user", "content": user_prompt},
    ]

def get_response(client: OpenAI, model: str, messages: list[ChatCompletionMessageParam]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    # We still want empty text to be returned and investigate later
    if not response.choices[0].message.content:
        return ""
    
    return response.choices[0].message.content

def clean_file(file_path: Path, client: OpenAI, model: str) -> str:
    with open(file_path.as_posix(), 'r', encoding='utf-8') as f:
        file_content = f.read()

    messages = get_messages(file_content)
    cleaned_content = get_response(client, model, messages)

    with open("cleaned_content.txt", 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    return cleaned_content

if __name__ == "__main__" :
    test_file = TXT_DIR / "test/code_du_travail_part_13_without_clean.txt"

    # MINISTRAL 14B works well
    # MISTRAL MEDIUM works well, but a bit less good
    # MISTRAL SMALL less good
    clean_file(test_file, frontier_mistral_client, MINISTRAL_14B)
