import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

from src.models.utils import test_model

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

GPT_5_NANO: Literal["gpt-5-nano"] = 'gpt-5-nano'
GPT_5_MINI: Literal["gpt-5-mini"] = 'gpt-5-mini'

GPT_4_1_MINI: Literal["gpt-4.1-mini"] = "gpt-4.1-mini"
GPT_4_1_NANO: Literal["gpt-4.1-nano"] = "gpt-4.1-nano"

GPT_4_O_MINI: Literal["gpt-4o-mini"] = "gpt-4o-mini"

client = OpenAI()

if __name__ == '__main__':
    model_to_test = GPT_4_1_MINI
    test_model(model_to_test, client, "Hello, how are you?")
    