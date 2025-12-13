"""
Define available mistrals models.
These models are available on the Ollama platform, and compatible with my NITRO5 laptop.

More models here: https://ollama.com/library/mistral
"""
from openai import OpenAI
from typing import Literal
from src.config import OLLAMA_URL
from src.models.utils import test_model

#  The 7B model released by Mistral AI, updated to version 0.3. 
# https://ollama.com/library/mistral
MISTRAL_7B: Literal["mistral:7b"] = "mistral:7b"

# A state-of-the-art 12B model with 128k context length, built by Mistral AI in collaboration with NVIDIA.
# https://ollama.com/library/mistral-nemo
MISTRAL_NEMO_12B: Literal["mistral-nemo:12b"] = "mistral-nemo:12b"


# MistralLite is a fine-tuned model based on Mistral with enhanced capabilities of processing long contexts. 
# https://ollama.com/library/mistrallite
MISTRAL_7B_LITE: Literal["mistrallite:7b"] = "mistrallite:7b"

# The Ministral 3 family is designed for edge deployment, capable of running on a wide range of hardware. 
# https://ollama.com/library/ministral-3
MINISTRAL3_3B: Literal["ministral-3:3b"] = "ministral-3:3b"

LLAMA_3_2: Literal["llama3.2:latest"] = "llama3.2:latest"

ollama_client = OpenAI(
    base_url=OLLAMA_URL,
    api_key='ollama',  # Ollama does not require an API key, but OpenAI requires it
)

if __name__ == '__main__':
    model_to_test = MISTRAL_7B
    test_model(model_to_test, ollama_client, "Bonjour toi !")
    
