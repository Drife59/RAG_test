from src.config import TXT_DIR
from src.models.mistral_models import frontier_mistral_client, MINISTRAL_3B
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam



user_prompt = """
    Tu vas recevoir en context le code du travail français, un document juridique séparé en articles.
    Ta mission est de séparer chaque article de ce document.
    Rajoute "[START]" au dessus de chaque titre d'article.
    Conserve exactement le texte comme il est, rajoute juste les séparateurs "[START]".
    Garde 100% du contenu original, n'enlève rien. Garde aussi les sauts de lignes.
    Ne rajoute pas de commentaire dans la réponse, renvoie uniquement le contenu.

    Exemple:
    "
    Article L1111-1
 
    Les dispositions du présent livre sont applicables aux employeurs de droit privé ainsi qu'à leurs salariés.
    
    Elles sont également applicables au personnel des personnes publiques employé dans les conditions du
    droit privé, sous réserve des dispositions particulières ayant le même objet résultant du statut qui régit ce
    personnel.
    "

    Devient après ton traitement:

    "
    [START]
    Article L1111-1
 
    Les dispositions du présent livre sont applicables aux employeurs de droit privé ainsi qu'à leurs salariés.
    
    Elles sont également applicables au personnel des personnes publiques employé dans les conditions du
    droit privé, sous réserve des dispositions particulières ayant le même objet résultant du statut qui régit ce
    personnel.
    "
    
    Contexte:
    {context} 
"""

def get_prompt(file_path: Path) -> str:
    with open(file_path.as_posix(), 'r', encoding='utf-8') as f:
        file_content = f.read()

        return user_prompt.format(context=file_content)
    
def get_message(prompt: str) -> ChatCompletionMessageParam:
    return {
        "role": "user",
        "content": prompt
    }
    
if __name__ == "__main__":
    test_file_path = TXT_DIR / "chunks/code_du_travail_part_1.txt"

    prompt = get_prompt(test_file_path)
    message = get_message(prompt)

    print(f"Start to mark articles on {test_file_path}...")
    response = frontier_mistral_client.chat.completions.create(
        model=MINISTRAL_3B,
        messages=[
            message
        ],
        temperature=0
    )

    destination_file_path = TXT_DIR / "marked_chunks/code_du_travail_part_1_marked.txt"

    with open(destination_file_path.as_posix(), 'w', encoding='utf-8') as f:
        if not response.choices[0].message.content:
            print("WARNING: LLM response is empty !")
        else:
            f.write(response.choices[0].message.content)
            print(f"Process done. File {destination_file_path} created.")