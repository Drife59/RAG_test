import pytest

from src.config import TXT_DIR
from src.models.mistral_models import MISTRAL_SMALL_32, frontier_mistral_client

# from src.models.ollama_models import ollama_client, MINISTRAL3_3B, MISTRAL_7B, LLAMA_3_2
from src.preprocessing.extractor.article_extractor import get_sourced_articles, index_article_by_id
from src.preprocessing.extractor.test.article_part.articles_part_13 import (
    CONTENT_ARTICLE_L1234_17,
    CONTENT_ARTICLE_L1234_17_1,
    CONTENT_ARTICLE_L1235_3,
    CONTENT_ARTICLE_L1235_17,
    CONTENT_ARTICLE_L1236_9,
    ID_ARTICLE_L1234_17,
    ID_ARTICLE_L1234_17_1,
    ID_ARTICLE_L1234_18,
    ID_ARTICLE_L1234_19,
    ID_ARTICLE_L1234_20,
    ID_ARTICLE_L1235_1,
    ID_ARTICLE_L1235_2,
    ID_ARTICLE_L1235_2_1,
    ID_ARTICLE_L1235_3,
    ID_ARTICLE_L1235_3_1,
    ID_ARTICLE_L1235_3_2,
    ID_ARTICLE_L1235_4,
    ID_ARTICLE_L1235_5,
    ID_ARTICLE_L1235_6,
    ID_ARTICLE_L1235_7,
    ID_ARTICLE_L1235_7_1,
    ID_ARTICLE_L1235_8,
    ID_ARTICLE_L1235_9,
    ID_ARTICLE_L1235_10,
    ID_ARTICLE_L1235_11,
    ID_ARTICLE_L1235_12,
    ID_ARTICLE_L1235_13,
    ID_ARTICLE_L1235_14,
    ID_ARTICLE_L1235_15,
    ID_ARTICLE_L1235_16,
    ID_ARTICLE_L1235_17,
    ID_ARTICLE_L1236_7,
    ID_ARTICLE_L1236_8,
    ID_ARTICLE_L1236_9,
)
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
    # (frontier_mistral_client, MINISTRAL_14B),
    
    # ~50 seconds
    (frontier_mistral_client, MISTRAL_SMALL_32),

    # ~60 seconds
    # (frontier_mistral_client, MISTRAL_MEDIUM_31),

    # ~80 seconds
    # (frontier_mistral_client, MISTRAL_LARGE_32),
]

@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_article_extractor_part_13(client, model):
    articles = get_sourced_articles(TEST_DIR / "code_du_travail_part_13.txt", client, model)

    assert len(articles) == 29

    article_by_id = index_article_by_id(articles)

    article_ids = article_by_id.keys()

    assert ID_ARTICLE_L1234_17 in article_ids
    assert ID_ARTICLE_L1234_17_1 in article_ids
    assert ID_ARTICLE_L1234_18 in article_ids
    assert ID_ARTICLE_L1234_19 in article_ids
    assert ID_ARTICLE_L1234_20 in article_ids
    assert ID_ARTICLE_L1235_1 in article_ids
    assert ID_ARTICLE_L1235_2 in article_ids
    assert ID_ARTICLE_L1235_2_1 in article_ids
    assert ID_ARTICLE_L1235_3 in article_ids
    assert ID_ARTICLE_L1235_3_1 in article_ids
    assert ID_ARTICLE_L1235_3_2 in article_ids
    assert ID_ARTICLE_L1235_4 in article_ids
    assert ID_ARTICLE_L1235_5 in article_ids
    assert ID_ARTICLE_L1235_6 in article_ids
    assert ID_ARTICLE_L1235_7 in article_ids
    assert ID_ARTICLE_L1235_7_1 in article_ids
    assert ID_ARTICLE_L1235_8 in article_ids
    assert ID_ARTICLE_L1235_9 in article_ids
    assert ID_ARTICLE_L1235_10 in article_ids
    assert ID_ARTICLE_L1235_11 in article_ids
    assert ID_ARTICLE_L1235_12 in article_ids
    assert ID_ARTICLE_L1235_13 in article_ids
    assert ID_ARTICLE_L1235_14 in article_ids
    assert ID_ARTICLE_L1235_15 in article_ids
    assert ID_ARTICLE_L1235_16 in article_ids
    assert ID_ARTICLE_L1235_17 in article_ids
    assert ID_ARTICLE_L1236_7 in article_ids
    assert ID_ARTICLE_L1236_8 in article_ids
    assert ID_ARTICLE_L1236_9 in article_ids

    print(article_by_id[ID_ARTICLE_L1235_3].content, flush=True)

    # This article has a very bad pdf transcription, allow bad similarity
    assert string_similarity(article_by_id[ID_ARTICLE_L1234_17].content, CONTENT_ARTICLE_L1234_17) > MINIMUM_SIMILARITY
    assert string_similarity(
        article_by_id[ID_ARTICLE_L1234_17_1].content, CONTENT_ARTICLE_L1234_17_1
    ) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1235_3].content, CONTENT_ARTICLE_L1235_3) > 0.7
    assert string_similarity(article_by_id[ID_ARTICLE_L1235_17].content, CONTENT_ARTICLE_L1235_17) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1236_9].content, CONTENT_ARTICLE_L1236_9) > MINIMUM_SIMILARITY
