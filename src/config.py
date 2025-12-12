from pathlib import Path

ANSWER_MODEL = "gpt-4.1-nano"

BASE_PATH = Path.cwd().resolve()
KNOWLEDGE_BASE_PATH = BASE_PATH / "knowledge-base"

DB_PATH = BASE_PATH / "vector_db"

PDF_DIR = KNOWLEDGE_BASE_PATH / "pdf"
TXT_DIR = KNOWLEDGE_BASE_PATH / "txt"
CODE_DU_TRAVAIL_PATH = PDF_DIR / "code_du_travail_7mo.pdf"
CONVENTION_COLLECTIVE_PATH = PDF_DIR / "convention_collective_syntec.pdf"

# TODO: have a centered embedding / retrieval model

# Number of chunk to retrieve to build answer
RETRIEVAL_K = 10

ANSWER_SYSTEM_PROMPT = """
Tu es un assistant pour répondre aux questions juridiques sur le code du travail.
Tu parles à des RH et des juristes.
Si c'est pertinent, utilise le contexte fourni pour répondre aux questions.
Si tu ne connais pas la réponse, dis le.
Contexte:
{context}
"""
