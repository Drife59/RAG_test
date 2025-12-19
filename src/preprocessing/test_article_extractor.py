from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_3B
from src.preprocessing.article_extractor import get_articles, index_article_by_id
from src.preprocessing.articles_part_1 import (
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


def string_similarity(chaine1: str, chaine2: str) -> float:
    def distance_levenshtein(s1, s2):
        if len(s1) < len(s2):
            return distance_levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    distance = distance_levenshtein(chaine1, chaine2)
    max_length = max(len(chaine1), len(chaine2))
    similarite = 1.0 - (distance / max_length) if max_length != 0 else 1.0

    return similarite

def test_article_extractor_part_1():
    articles = get_articles(TEST_DIR / "code_du_travail_part_1.txt", MINISTRAL_3B)
    # 13 articles in this first file, but the last one is troncated and was removed
    assert len(articles) == 12

    assert articles[0]['id'] == ID_ARTICLE_L1
    assert string_similarity(articles[0]['content'], CONTENT_ARTICLE_L1) > 0.97

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

    assert string_similarity(article_by_id[ID_ARTICLE_L1], CONTENT_ARTICLE_L1) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L2], CONTENT_ARTICLE_L2) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L3], CONTENT_ARTICLE_L3) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_1], CONTENT_ARTICLE_L1111_1) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_2], CONTENT_ARTICLE_L1111_2) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1111_3], CONTENT_ARTICLE_L1111_3) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_1], CONTENT_ARTICLE_L1121_1) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1121_2], CONTENT_ARTICLE_L1121_2) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1131_1], CONTENT_ARTICLE_L1131_1) > 0.97
    # Does not work because of the "chapitre II (...)"
    # assert string_similarity(article_by_id[ID_ARTICLE_L1131_2], CONTENT_ARTICLE_L1131_2) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1132_1], CONTENT_ARTICLE_L1132_1) > 0.97
    assert string_similarity(article_by_id[ID_ARTICLE_L1132_2], CONTENT_ARTICLE_L1132_2) > 0.97

    

