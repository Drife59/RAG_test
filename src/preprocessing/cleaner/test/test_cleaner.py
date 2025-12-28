import pytest

from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_14B, frontier_mistral_client
from src.preprocessing.cleaner.cleaner import clean_file
from src.preprocessing.utils import string_similarity

TEST_DIR = TXT_DIR / "test" 

# Removing noise (section, chapter, title, etc.) should not modify a lot the content
MINIMUM_SIMILARITY = 0.95

MODELS_TO_TEST = [
    # -----  WORKING MODELS --------

    # (frontier_mistral_client, MINISTRAL_3B),
    # (frontier_mistral_client, MINISTRAL_8B),
    (frontier_mistral_client, MINISTRAL_14B),
    
    # (frontier_mistral_client, MISTRAL_SMALL_32),

    # (frontier_mistral_client, MISTRAL_MEDIUM_31),

    # (frontier_mistral_client, MISTRAL_LARGE_32),
]

@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_article_extractor_part_1(client, model):
    cleaned_content = clean_file(TEST_DIR / "code_du_travail_part_13_without_clean.txt", client, model)

    with open(TEST_DIR / "code_du_travail_part_13.txt", 'r', encoding='utf-8') as f:
        expected_content = f.read()

    assert string_similarity(expected_content, cleaned_content) > MINIMUM_SIMILARITY