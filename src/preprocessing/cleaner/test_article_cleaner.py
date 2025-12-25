import pytest

# from src.models.mistral_models import frontier_mistral_client, MINISTRAL_3B as FRONTIER_MINISTRAL_3B
from src.models.ollama_models import MINISTRAL3_3B, ollama_client
from src.preprocessing.cleaner.article_cleaner import clean_article_content
from src.preprocessing.cleaner.content_article_cleaner import (
    article_with_chapitre,
    article_with_noise_both_side,
    article_with_plenty_noise,
    cleaned_article_with_chapitre,
    cleaned_article_with_noise_both_side,
    cleaned_article_with_plenty_noise,
)
from src.preprocessing.utils import string_similarity

MINIMUM_SIMILARITY = 0.95

MODELS_TO_TEST = [
    (ollama_client, MINISTRAL3_3B),
    # (frontier_mistral_client, FRONTIER_MINISTRAL_3B),
]

@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_clean_article_with_chapitre(client, model):
    cleaned_article = clean_article_content(client, model, article_with_chapitre)
    assert string_similarity(cleaned_article, cleaned_article_with_chapitre) > MINIMUM_SIMILARITY

@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_clean_article_with_plenty_noise(client, model):
    cleaned_article = clean_article_content(client, model, article_with_plenty_noise)
    assert string_similarity(cleaned_article, cleaned_article_with_plenty_noise) > MINIMUM_SIMILARITY

@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_article_with_noise_both_side(client, model):
    cleaned_article = clean_article_content(ollama_client, MINISTRAL3_3B, article_with_noise_both_side)
    assert string_similarity(cleaned_article, cleaned_article_with_noise_both_side) > MINIMUM_SIMILARITY

    

