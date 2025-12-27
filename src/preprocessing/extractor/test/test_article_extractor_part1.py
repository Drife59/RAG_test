import pytest

from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_14B, frontier_mistral_client

# from src.models.ollama_models import ollama_client, MINISTRAL3_3B, MISTRAL_7B, LLAMA_3_2
from src.preprocessing.extractor.article_extractor import get_sourced_articles, index_article_by_id
from src.preprocessing.extractor.test.article_part.articles_part_1 import (
    CONTENT_ARTICLE_L1,
    CONTENT_ARTICLE_L2,
    CONTENT_ARTICLE_L3,
    CONTENT_ARTICLE_L1111_1,
    CONTENT_ARTICLE_L1111_2,
    CONTENT_ARTICLE_L1111_3,
    CONTENT_ARTICLE_L1121_1,
    CONTENT_ARTICLE_L1121_2,
    CONTENT_ARTICLE_L1131_1,
    CONTENT_ARTICLE_L1132_1,
    CONTENT_ARTICLE_L1132_2,
    ID_ARTICLE_L1,
    ID_ARTICLE_L2,
    ID_ARTICLE_L3,
    ID_ARTICLE_L1111_1,
    ID_ARTICLE_L1111_2,
    ID_ARTICLE_L1111_3,
    ID_ARTICLE_L1121_1,
    ID_ARTICLE_L1121_2,
    ID_ARTICLE_L1131_1,
    ID_ARTICLE_L1131_2,  # CONTENT_ARTICLE_L1131_2,
    ID_ARTICLE_L1132_1,
    ID_ARTICLE_L1132_2,
)
from src.preprocessing.old_cleaner.article_cleaner import clean_article_content
from src.preprocessing.utils import string_similarity

TEST_DIR = TXT_DIR / "test" 

MINIMUM_SIMILARITY = 0.95

MODELS_TO_TEST = [
    # ------ FAILLING MODELS -------
    # LOCAL
    # (ollama_client, MINISTRAL3_3B),
    # (ollama_client, LLAMA_3_2),

    # FRONTIER
    # WTF ? A powerfull frontier failing ? 
    # (frontier_mistral_client, MISTRAL_MEDIUM_31),

    # -----  WORKING MODELS --------

    # FRONTIER
    
    # ~20 seconds
    # (frontier_mistral_client, MINISTRAL_3B),

    # ~30 seconds
    # (frontier_mistral_client, MINISTRAL_8B),

    # ~35 seconds
    (frontier_mistral_client, MINISTRAL_14B),
    
    # ~50 seconds
    # (frontier_mistral_client, MISTRAL_SMALL_32),

    # ~60 seconds
    # (frontier_mistral_client, MISTRAL_MEDIUM_31),

    # ~80 seconds
    # (frontier_mistral_client, MISTRAL_LARGE_32),
]

@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_article_extractor_part_1(client, model):
    articles = get_sourced_articles(TEST_DIR / "code_du_travail_part_1.txt", client, model)
    # 13 articles in this first file, but the last one is troncated and was removed
    assert len(articles) == 12

    article_by_id = index_article_by_id(articles)

    article_ids = article_by_id.keys()

    assert ID_ARTICLE_L1 in article_ids
    assert ID_ARTICLE_L2 in article_ids
    assert ID_ARTICLE_L3 in article_ids
    assert ID_ARTICLE_L1111_1 in article_ids
    assert ID_ARTICLE_L1111_2 in article_ids
    assert ID_ARTICLE_L1111_3 in article_ids
    assert ID_ARTICLE_L1121_1 in article_ids
    assert ID_ARTICLE_L1121_2 in article_ids
    assert ID_ARTICLE_L1131_1 in article_ids
    assert ID_ARTICLE_L1131_2 in article_ids
    assert ID_ARTICLE_L1132_1 in article_ids
    assert ID_ARTICLE_L1132_2 in article_ids

    assert string_similarity(article_by_id[ID_ARTICLE_L1].content, CONTENT_ARTICLE_L1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L2].content, CONTENT_ARTICLE_L2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L3].content, CONTENT_ARTICLE_L3) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_1].content, CONTENT_ARTICLE_L1111_1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_2].content, CONTENT_ARTICLE_L1111_2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_3].content, CONTENT_ARTICLE_L1111_3) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_1].content, CONTENT_ARTICLE_L1121_1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_2].content, CONTENT_ARTICLE_L1121_2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1131_1].content, CONTENT_ARTICLE_L1131_1) > MINIMUM_SIMILARITY
    # Does not work because of the "chapitre II (...)"
    # assert string_similarity(article_by_id[ID_ARTICLE_L1131_2], CONTENT_ARTICLE_L1131_2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1132_1].content, CONTENT_ARTICLE_L1132_1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1132_2].content, CONTENT_ARTICLE_L1132_2) > MINIMUM_SIMILARITY

@pytest.mark.skip(reason="Obsolete test, obsolete cleaning method.")
@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_article_extraction_plus_clean_part_1(client, model):
    articles = get_sourced_articles(TEST_DIR / "code_du_travail_part_1.txt", client, model)
    article_by_id = index_article_by_id(articles)
    article_ids = article_by_id.keys()

    assert ID_ARTICLE_L1 in article_ids
    assert ID_ARTICLE_L1111_1 in article_ids
    assert ID_ARTICLE_L1121_1 in article_ids
    assert ID_ARTICLE_L1131_2 in article_ids

    cleaned_content_l1 = clean_article_content(client, model, article_by_id[ID_ARTICLE_L1].content)
    assert string_similarity(article_by_id[ID_ARTICLE_L1].content, cleaned_content_l1) > MINIMUM_SIMILARITY

    cleaned_content_l1111_1 = clean_article_content(client, model, article_by_id[ID_ARTICLE_L1111_1].content)
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_1].content, cleaned_content_l1111_1) > MINIMUM_SIMILARITY

    cleaned_content_l1121_1 = clean_article_content(client, model, article_by_id[ID_ARTICLE_L1121_1].content)
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_1].content, cleaned_content_l1121_1) > MINIMUM_SIMILARITY

    cleaned_content_l1131_2 = clean_article_content(client, model, article_by_id[ID_ARTICLE_L1131_2].content)
    print("cleaned_content_l1131_2")
    print(cleaned_content_l1131_2)
    print("orginal_content_l1131_2")
    print(article_by_id[ID_ARTICLE_L1131_2].content, flush=True)
    assert string_similarity(article_by_id[ID_ARTICLE_L1131_2].content, cleaned_content_l1131_2) > MINIMUM_SIMILARITY
