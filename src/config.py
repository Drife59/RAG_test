from pathlib import Path

ANSWER_MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
# Number of chunk to retrieve to build answer
RETRIEVAL_K = 20

ANSWER_SYSTEM_PROMPT = """
Tu es un assistant pour répondre aux questions juridiques sur le code du travail.
Tu parles à des RH et des juristes.
Si c'est pertinent, utilise le contexte fourni pour répondre aux questions.
Si tu ne connais pas la réponse, dis le.
Contexte:
{context}
"""
