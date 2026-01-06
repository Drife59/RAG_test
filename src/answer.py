import sys
import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import ANSWER_MODEL, ANSWER_SYSTEM_PROMPT, DB_PATH, RETRIEVAL_K
from src.embeddings.embedding_models import current_embedding_model
from src.inference.reranking import get_filtered_contexts

# We need to do this trick, since python until 3.14 has sqlite3 3.31
# but Chroma requires 3.35+
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from dotenv import load_dotenv  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

load_dotenv(override=True)

vectorstore = Chroma(persist_directory=DB_PATH.as_posix(), embedding_function=current_embedding_model)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model=ANSWER_MODEL)


def fetch_context(question: str, rerank_filter: bool = True) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    context_docs = retriever.invoke(question, k=RETRIEVAL_K)

    if not rerank_filter:
        return context_docs

    start_time = time.time()
    filtered_context_docs = get_filtered_contexts(question, context_docs)
    end_time = time.time()

    print(f"time to validate context: {end_time - start_time}")

    return filtered_context_docs


# This combined context is from the udemy course.
# combined = combined_question(question, history)
# For the "code du travail", actually combining question degrade the quality of the answer.
def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def get_system_prompt(context_docs: list[Document]) -> str:
    context = "\n\n".join(context_doc.page_content for context_doc in context_docs)
    return ANSWER_SYSTEM_PROMPT.format(context=context)


def get_messages(system_prompt: str, history: list[dict], question: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    return messages


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    context_docs = fetch_context(question)
    system_prompt = get_system_prompt(context_docs)
    messages = get_messages(system_prompt, history, question)

    response = llm.invoke(messages)

    # I'm unsure why response.content is not a always a string. Just make it that way.
    response_content: str = str(response.content)
    return response_content, context_docs


if __name__ == "__main__":
    question = "Quels sont les codes APE qui relèvent de la convention syntec ?"
    answer, docs = answer_question(question)
    print("Question utilisateur:")
    print(question)
    print("\nReponse:")
    print(answer)
