from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"

class CustomSentenceTransformerEmbeddings(Embeddings):
    """
        Implement langchain Embedding interface using sentence transformer for model.

        Allow to use a model compatible with SentenceTransformer but not
        officially supported by Langchain.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL, trust_remote_code: bool = False):
        self.model = SentenceTransformer(model_name, device="cuda", trust_remote_code=trust_remote_code)
        self.model.half()  # Quantitize the model by 50%

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=4,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            [text],
            batch_size=1,
            convert_to_tensor=False
        )
        return embedding.tolist()[0]
