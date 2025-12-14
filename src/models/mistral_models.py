import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal
from src.models.utils import test_model

load_dotenv(override=True)
mistral_api_key = os.getenv('MISTRAL_API_KEY')

MINISTRAL_8B: Literal["ministral-8b-2512"] = 'ministral-8b-2512'
MINISTRAL_3B: Literal["ministral-3b-2512"] = 'ministral-3b-2512'
MISTRAL_SMALL_32: Literal["mistral-small-2506"] = 'mistral-small-2506'
MISTRAL_MEDIUM_31: Literal["mistral-medium-2508"] = 'mistral-medium-2508'
MAGISTRAL_MEDIUM_12: Literal["magistral-medium-2509"] = 'magistral-medium-2509'


frontier_mistral_client = OpenAI(
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1/"
)

if __name__ == '__main__':
    model_to_test = MINISTRAL_3B
    test_model(model_to_test, frontier_mistral_client, "Hello, how are you?")