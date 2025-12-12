# from langchain_huggingface import HuggingFaceEmbeddings

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
# "Please pass the argument `trust_remote_code=True` to allow custom code to be run."
# DANG_FRENCH_DOCUMENT = "dangvantuan/french-document-embedding"

# Does not work. Some bug within sentence_transformers
# https://huggingface.co/dangvantuan/sentence-camembert-large/discussions/12
# DANG_CAMEMBERT_LARGE_S = "dangvantuan/sentence-camembert-large"
# Works with NASTY warnings ("you shoud probably train this model....") 
# DANG_CAMEMBERT_LARGE_C = "dangvantuan/CrossEncoder-camembert-large"

# This 2 models below does not work at all
# OSError: almanach/moderncamembert-large-ckpts does not appear to have a file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt or flax_model.msgpack.
# ALMANACH_CAMEMBERT_BASE_CKPTS = "almanach/moderncamembert-base-ckpts"
# ALMANACH_CAMEMBERT_LARGE = "almanach/moderncamembert-large-ckpts"

# 2 models below works but with a warning... Are results good ? 
# "No sentence-transformers model found with name almanach/moderncamembert-base. Creating a new one with mean pooling."
# ALMANACH_CAMEMBERT_BASE = "almanach/moderncamembert-base"
# ALMANACH_CAMEMBERT_BASE_CV2 = "almanach/moderncamembert-cv2-base"

# Multi langage, 1024 dim vectors
INTFLOAT_MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"

# EuroBERT are europeen models
# https://huggingface.co/EuroBERT
# EUROBERT_210M = "EuroBERT/EuroBERT-210m"
EUROBERT_610M = "EuroBERT/EuroBERT-610m"
# EUROBERT_2G = "EuroBERT/EuroBERT-2G"

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

