from src.config import TXT_DIR
from src.preprocessing.utils import string_similarity
from src.models.mistral_models import MINISTRAL_3B
from src.preprocessing.article_extractor import get_articles, index_article_by_id
from src.preprocessing.article_part.articles_part_1 import (
    ID_ARTICLE_L1, CONTENT_ARTICLE_L1,
    ID_ARTICLE_L2, CONTENT_ARTICLE_L2,
    ID_ARTICLE_L3, CONTENT_ARTICLE_L3,
    ID_ARTICLE_L1111_1, CONTENT_ARTICLE_L1111_1,
    ID_ARTICLE_L1111_2, CONTENT_ARTICLE_L1111_2,
    ID_ARTICLE_L1111_3, CONTENT_ARTICLE_L1111_3,
    ID_ARTICLE_L1121_1, CONTENT_ARTICLE_L1121_1,
    ID_ARTICLE_L1121_2, CONTENT_ARTICLE_L1121_2,
    ID_ARTICLE_L1131_1, CONTENT_ARTICLE_L1131_1,
    ID_ARTICLE_L1131_2, CONTENT_ARTICLE_L1131_2,
    ID_ARTICLE_L1132_1, CONTENT_ARTICLE_L1132_1,
    ID_ARTICLE_L1132_2, CONTENT_ARTICLE_L1132_2
)

TEST_DIR = TXT_DIR / "raw_chunks" 

MINIMUM_SIMILARITY = 0.97

def test_article_extractor_part_1():
    articles = get_articles(TEST_DIR / "code_du_travail_part_1.txt", MINISTRAL_3B)
    # 13 articles in this first file, but the last one is troncated and was removed
    assert len(articles) == 12

    assert articles[0]['id'] == ID_ARTICLE_L1
    assert string_similarity(articles[0]['content'], CONTENT_ARTICLE_L1) > MINIMUM_SIMILARITY

    article_by_id = index_article_by_id(articles)

    article_ids = article_by_id.keys()

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

    assert string_similarity(article_by_id[ID_ARTICLE_L1], CONTENT_ARTICLE_L1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L2], CONTENT_ARTICLE_L2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L3], CONTENT_ARTICLE_L3) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_1], CONTENT_ARTICLE_L1111_1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_2], CONTENT_ARTICLE_L1111_2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_3], CONTENT_ARTICLE_L1111_3) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_1], CONTENT_ARTICLE_L1121_1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_2], CONTENT_ARTICLE_L1121_2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1131_1], CONTENT_ARTICLE_L1131_1) > MINIMUM_SIMILARITY
    # Does not work because of the "chapitre II (...)"
    # assert string_similarity(article_by_id[ID_ARTICLE_L1131_2], CONTENT_ARTICLE_L1131_2) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1132_1], CONTENT_ARTICLE_L1132_1) > MINIMUM_SIMILARITY
    assert string_similarity(article_by_id[ID_ARTICLE_L1132_2], CONTENT_ARTICLE_L1132_2) > MINIMUM_SIMILARITY

    

