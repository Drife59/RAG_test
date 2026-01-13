import sys
import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import ANSWER_MODEL, ANSWER_SYSTEM_PROMPT, DB_PATH, RETRIEVAL_K
from src.embeddings.embedding_models import current_embedding_model
from src.inference.reranking import get_filtered_contexts
from src.utils.utils import display_execution_time

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


@display_execution_time
def fetch_context(question: str, rerank_filter: bool = True) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    As rerank is very time consuming (12-15 seconds for MISTRAL_SMALL_32), we can disable it.
    """
    context_docs = retriever.invoke(question, k=RETRIEVAL_K)

    if not rerank_filter:
        return context_docs

    start_time = time.time()
    filtered_context_docs = get_filtered_contexts(question, context_docs)
    print(f"[fetch_context] time to validate context: {time.time() - start_time}")

    return filtered_context_docs


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
    Answer the given question with RAG and optionnal rerank filter.
    Return the answer and the context documents.
    """
    context_docs = fetch_context(question)
    system_prompt = get_system_prompt(context_docs)
    messages = get_messages(system_prompt, history, question)

    response = llm.invoke(messages)

    # I'm unsure why response.content is not a always a string. Just make it that way.
    response_content: str = str(response.content)
    return response_content, context_docs
