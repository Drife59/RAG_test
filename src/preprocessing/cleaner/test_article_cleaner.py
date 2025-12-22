from src.preprocessing.cleaner.article_cleaner import clean_article
from src.preprocessing.cleaner.content_article_cleaner import (
    article_with_chapitre,
    cleaned_article_with_chapitre,
    article_with_plenty_noise,
    cleaned_article_with_plenty_noise,
    article_with_noise_both_side,
    cleaned_article_with_noise_both_side
)
from src.preprocessing.utils import string_similarity
from src.models.mistral_models import frontier_mistral_client

MINIMUM_SIMILARITY = 0.95

def test_clean_article_with_chapitre():
    cleaned_article = clean_article(frontier_mistral_client, article_with_chapitre)
    assert string_similarity(cleaned_article, cleaned_article_with_chapitre) > MINIMUM_SIMILARITY

def test_clean_article_with_plenty_noise():
    cleaned_article = clean_article(frontier_mistral_client, article_with_plenty_noise)
    assert string_similarity(cleaned_article, cleaned_article_with_plenty_noise) > MINIMUM_SIMILARITY

def test_article_with_noise_both_side():
    cleaned_article = clean_article(frontier_mistral_client, article_with_noise_both_side)
    assert string_similarity(cleaned_article, cleaned_article_with_noise_both_side) > MINIMUM_SIMILARITY

    

