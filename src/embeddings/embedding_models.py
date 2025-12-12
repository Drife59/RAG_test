from dataclasses import dataclass
from typing import Literal, Type
from langchain_huggingface import HuggingFaceEmbeddings

from src.embeddings.customTansformer import CustomSentenceTransformerEmbeddings

# Available native ingest model

# ---------------
# LOCAL EXECUTION
# ---------------

# Theses models will leverage local GPU and CPU

# English, 384 dim vectors  
HUGGINGFACE_MINILM_L6 = "all-MiniLM-L6-v2"
# English, 768 dim vectors
BGE_BASE_EN = "BAAI/bge-base-en-v1.5"
# English, 1024 dim vectors 
BGE_LARGE_EN = "BAAI/bge-large-en-v1.5" # Warning: ingestion can very long
# multilingual, 1024 dim vectors
BGE_M3 = "BAAI/bge-m3"

# French, 768 dim vectors
DANG_CAMEMBERT_BASE = "dangvantuan/sentence-camembert-base"
# Does not work: 
DANG_FRENCH_DOCUMENT = "dangvantuan/french-document-embedding"

# Works with NASTY warnings ("you shoud probably train this model....") 
DANG_CAMEMBERT_LARGE_C = "dangvantuan/CrossEncoder-camembert-large"

# Multi langage, 1024 dim vectors
INTFLOAT_MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"

# EuroBERT are europeen models, trained by UE !
# https://huggingface.co/EuroBERT
# Works with a warning: 
# No sentence-transformers model found with name EuroBERT/EuroBERT-610m. Creating a new one with mean pooling.
EUROBERT_210M = "EuroBERT/EuroBERT-210m"
EUROBERT_610M = "EuroBERT/EuroBERT-610m"
EUROBERT_2G = "EuroBERT/EuroBERT-2G"

available_models = Literal[
    "all-MiniLM-L6-v2", 
    "BAAI/bge-base-en-v1.5", 
    "BAAI/bge-large-en-v1.5", 
    "BAAI/bge-m3", 
    "dangvantuan/sentence-camembert-base", 
    "dangvantuan/french-document-embedding", 
    "dangvantuan/CrossEncoder-camembert-large",
    "intfloat/multilingual-e5-large",
    "EuroBERT/EuroBERT-210m", 
    "EuroBERT/EuroBERT-610m", 
    "EuroBERT/EuroBERT-2G"
]

@dataclass
class LangchainModelConfig:
    model: Type[HuggingFaceEmbeddings | CustomSentenceTransformerEmbeddings]
    # If the model need trust_remote_code to be true
    trust_remote_code: bool = False


# We don't want to instanciate the model here, it would fill up the GPU memory
local_model_to_config: dict[str, LangchainModelConfig] = {
    HUGGINGFACE_MINILM_L6: LangchainModelConfig(HuggingFaceEmbeddings, trust_remote_code=False),
    BGE_BASE_EN: LangchainModelConfig(HuggingFaceEmbeddings, trust_remote_code=False),
    BGE_LARGE_EN: LangchainModelConfig(HuggingFaceEmbeddings, trust_remote_code=False),
    BGE_M3: LangchainModelConfig(HuggingFaceEmbeddings, trust_remote_code=False),
    DANG_CAMEMBERT_BASE: LangchainModelConfig(CustomSentenceTransformerEmbeddings, trust_remote_code=False),
    DANG_FRENCH_DOCUMENT: LangchainModelConfig(CustomSentenceTransformerEmbeddings, trust_remote_code=False),
    DANG_CAMEMBERT_LARGE_C:LangchainModelConfig(CustomSentenceTransformerEmbeddings, trust_remote_code=False),
    EUROBERT_210M: LangchainModelConfig(CustomSentenceTransformerEmbeddings, trust_remote_code=False),
    EUROBERT_610M: LangchainModelConfig(CustomSentenceTransformerEmbeddings, trust_remote_code=False),
    EUROBERT_2G: LangchainModelConfig(CustomSentenceTransformerEmbeddings, trust_remote_code=False),
}

# -------------------------
# EMBEDDING MODEL INSTANCES
# -------------------------

# Define here instances, ready to use with Langchain / Chroma.

# hugging_face_minilm6_embeddings = HuggingFaceEmbeddings(model=HUGGINGFACE_MINILM_L6)
# hugging_face_bge_base_en_embeddings = HuggingFaceEmbeddings(model=BGE_BASE_EN)
# hugging_face_bge_large_en_embeddings = HuggingFaceEmbeddings(model=BGE_LARGE_EN)

# dangvantuan_camembert_base_embeddings = CustomSentenceTransformerEmbeddings(model_name=DANG_CAMEMBERT_BASE)
# eurobert_210M_embeddings = CustomSentenceTransformerEmbeddings(model_name=EUROBERT_210M, trust_remote_code=True)
eurobert_610M_embeddings = CustomSentenceTransformerEmbeddings(model_name=EUROBERT_610M, trust_remote_code=True)
# eurobert_2G_embeddings = CustomSentenceTransformerEmbeddings(model_name=EUROBERT_2G, trust_remote_code=True)
# intfloat_multilingual_e5_large_embeddings = CustomSentenceTransformerEmbeddings(model_name=INTFLOAT_MULTILINGUAL_E5_LARGE)

