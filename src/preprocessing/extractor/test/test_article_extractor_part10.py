import pytest

from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_14B, frontier_mistral_client

# from src.models.ollama_models import ollama_client, MINISTRAL3_3B, MISTRAL_7B, LLAMA_3_2
from src.preprocessing.extractor.article_extractor import get_sourced_articles, index_article_by_id
from src.preprocessing.extractor.test.article_part.articles_part_10 import (
    CONTENT_ARTICLE_L1225_24,
    CONTENT_ARTICLE_L1225_25,
    CONTENT_ARTICLE_L1225_26,
    CONTENT_ARTICLE_L1225_27,
    CONTENT_ARTICLE_L1225_28,
    CONTENT_ARTICLE_L1225_29,
    CONTENT_ARTICLE_L1225_30,
    CONTENT_ARTICLE_L1225_31,
    CONTENT_ARTICLE_L1225_32,
    CONTENT_ARTICLE_L1225_33,
    CONTENT_ARTICLE_L1225_34,
    CONTENT_ARTICLE_L1225_35,
    CONTENT_ARTICLE_L1225_36,
    CONTENT_ARTICLE_L1225_37,
    CONTENT_ARTICLE_L1225_38,
    CONTENT_ARTICLE_L1225_39,
    CONTENT_ARTICLE_L1225_40,
    CONTENT_ARTICLE_L1225_41,
    ID_ARTICLE_L1225_24,
    ID_ARTICLE_L1225_25,
    ID_ARTICLE_L1225_26,
    ID_ARTICLE_L1225_27,
    ID_ARTICLE_L1225_28,
    ID_ARTICLE_L1225_29,
    ID_ARTICLE_L1225_30,
    ID_ARTICLE_L1225_31,
    ID_ARTICLE_L1225_32,
    ID_ARTICLE_L1225_33,
    ID_ARTICLE_L1225_34,
    ID_ARTICLE_L1225_35,
    ID_ARTICLE_L1225_36,
    ID_ARTICLE_L1225_37,
    ID_ARTICLE_L1225_38,
    ID_ARTICLE_L1225_39,
    ID_ARTICLE_L1225_40,
    ID_ARTICLE_L1225_41,
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
    (frontier_mistral_client, MINISTRAL_14B),
    # ~50 seconds
    # (frontier_mistral_client, MISTRAL_SMALL_32),
    # ~60 seconds
    # (frontier_mistral_client, MISTRAL_MEDIUM_31),
    # ~80 seconds
    # (frontier_mistral_client, MISTRAL_LARGE_32),
]


@pytest.mark.parametrize("client, model", MODELS_TO_TEST)
def test_article_extractor_part_10(client, model):
    articles = get_sourced_articles(TEST_DIR / "code_du_travail_part_10.txt", client, model)

    assert len(articles) == 20

    article_by_id = index_article_by_id(articles)

    article_ids = article_by_id.keys()

    assert ID_ARTICLE_L1225_24 in article_ids
    assert ID_ARTICLE_L1225_25 in article_ids
    assert ID_ARTICLE_L1225_26 in article_ids
    assert ID_ARTICLE_L1225_27 in article_ids
    assert ID_ARTICLE_L1225_28 in article_ids
    assert ID_ARTICLE_L1225_29 in article_ids
    assert ID_ARTICLE_L1225_30 in article_ids
    assert ID_ARTICLE_L1225_31 in article_ids
    assert ID_ARTICLE_L1225_32 in article_ids
    assert ID_ARTICLE_L1225_33 in article_ids
    assert ID_ARTICLE_L1225_34 in article_ids
    assert ID_ARTICLE_L1225_35 in article_ids
    assert ID_ARTICLE_L1225_36 in article_ids
    assert ID_ARTICLE_L1225_37 in article_ids
    assert ID_ARTICLE_L1225_38 in article_ids
    assert ID_ARTICLE_L1225_39 in article_ids
    assert ID_ARTICLE_L1225_40 in article_ids
    assert ID_ARTICLE_L1225_41 in article_ids

    assert string_similarity(article_by_id[ID_ARTICLE_L1225_24].content, CONTENT_ARTICLE_L1225_24) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_25].content, CONTENT_ARTICLE_L1225_25) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_26].content, CONTENT_ARTICLE_L1225_26) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_27].content, CONTENT_ARTICLE_L1225_27) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_28].content, CONTENT_ARTICLE_L1225_28) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_29].content, CONTENT_ARTICLE_L1225_29) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_30].content, CONTENT_ARTICLE_L1225_30) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_31].content, CONTENT_ARTICLE_L1225_31) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_32].content, CONTENT_ARTICLE_L1225_32) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_33].content, CONTENT_ARTICLE_L1225_33) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_34].content, CONTENT_ARTICLE_L1225_34) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_35].content, CONTENT_ARTICLE_L1225_35) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_36].content, CONTENT_ARTICLE_L1225_36) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_37].content, CONTENT_ARTICLE_L1225_37) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_38].content, CONTENT_ARTICLE_L1225_38) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_39].content, CONTENT_ARTICLE_L1225_39) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_40].content, CONTENT_ARTICLE_L1225_40) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1225_41].content, CONTENT_ARTICLE_L1225_41) > MINIMUM_SIMILARITY
