# import pytest
from pathlib import Path
# from src.config import TXT_DIR
# from src.models.mistral_models import MINISTRAL_3B
from src.preprocessing.article_extractor import get_articles
TXT_DIR = Path.cwd().resolve() / "knowledge-base" / "txt" / "raw_chunks" 


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

def test_article_extractor_part_1():
    articles = get_articles(TXT_DIR / "code_du_travail_part_1.txt", 'ministral-3b-2512')
    # 13 articles in this first file, but the last one is troncated and was removed
    assert len(articles) == 12

    assert articles[0]['id'] == ID_ARTICLE_L1
    # assert articles[0]['content'].strip() == CONTENT_ARTICLE_L1.strip()