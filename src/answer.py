import sys
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.messages.base import BaseMessage
from langchain_core.documents import Document
from src.embeddings.embedding_models import INTFLOAT_MULTILINGUAL_E5_LARGE, langchain_embedding_model_factory
from src.config import ANSWER_MODEL, DB_PATH, RETRIEVAL_K, ANSWER_SYSTEM_PROMPT

# We need to do this trick, since python until 3.14 has sqlite3 3.31
# but Chroma requires 3.35+
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma # noqa: E402

from dotenv import load_dotenv # noqa: E402

load_dotenv(override=True)

embeddings = langchain_embedding_model_factory(INTFLOAT_MULTILINGUAL_E5_LARGE)

vectorstore = Chroma(persist_directory=DB_PATH.as_posix(), embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model=ANSWER_MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = ANSWER_SYSTEM_PROMPT.format(context=context)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    # I'm unsure why response.content is not a always a string. Just make it that way.
    response_content: str = str(response.content)
    return response_content, docs


if __name__ == "__main__":
    question="Quels sont les codes APE qui relèvent de la convention syntec ?"
    answer, docs = answer_question(question)
    print("Question utilisateur:")
    print(question)
    print("\nReponse:")
    print(answer)