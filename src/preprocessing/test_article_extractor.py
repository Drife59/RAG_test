from src.config import TXT_DIR
from src.models.mistral_models import MINISTRAL_3B
from src.preprocessing.article_extractor import get_articles, index_article_by_id

TEST_DIR = TXT_DIR / "raw_chunks" 


ID_ARTICLE_L1 = "Article L1"
CONTENT_ARTICLE_L1 = """
Tout projet de réforme envisagé par le Gouvernement qui porte sur les relations individuelles et collectives
du travail, l'emploi et la formation professionnelle et qui relève du champ de la négociation nationale et
interprofessionnelle fait l'objet d'une concertation préalable avec les organisations syndicales de salariés et
d'employeurs représentatives au niveau national et interprofessionnel en vue de l'ouverture éventuelle d'une
telle négociation.
 
A cet effet, le Gouvernement leur communique un document d'orientation présentant des éléments de
diagnostic, les objectifs poursuivis et les principales options.
 
Lorsqu'elles font connaître leur intention d'engager une telle négociation, les organisations indiquent
également au Gouvernement le délai qu'elles estiment nécessaire pour conduire la négociation.
 
Le présent article n'est pas applicable en cas d'urgence. Lorsque le Gouvernement décide de mettre en
oeuvre un projet de réforme en l'absence de procédure de concertation, il fait connaître cette décision
aux organisations mentionnées au premier alinéa en la motivant dans un document qu'il transmet à ces
organisations avant de prendre toute mesure nécessitée par l'urgence.
"""

ID_ARTICLE_L1121_1 = "Article L1121-1"
CONTENT_ARTICLE_L1121_1 = """ 
Nul ne peut apporter aux droits des personnes et aux libertés individuelles et collectives de restrictions qui ne
seraient pas justifiées par la nature de la tâche à accomplir ni proportionnées au but recherché.
"""

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

    assert "Article L2" in article_ids
    assert "Article L3" in article_ids
    assert "Article L1111-1" in article_ids
    assert "Article L1111-2" in article_ids
    assert "Article L1111-3" in article_ids
    assert "Article L1121-1" in article_ids
    assert "Article L1121-2" in article_ids
    assert "Article L1131-1" in article_ids
    assert "Article L1131-2" in article_ids
    assert "Article L1132-1" in article_ids
    assert "Article L1132-2" in article_ids

    assert string_similarity(article_by_id[ID_ARTICLE_L1121_1], CONTENT_ARTICLE_L1121_1) > 0.97
    

