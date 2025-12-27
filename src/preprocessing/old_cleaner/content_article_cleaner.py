article_with_chapitre = """
Dans toute entreprise employant au moins trois cents salariés et dans toute entreprise spécialisée dans
le recrutement, les employés chargés des missions de recrutement reçoivent une formation à la non-
discrimination à l'embauche au moins une fois tous les cinq ans.
Chapitre II : Principe de non-discrimination.
"""

cleaned_article_with_chapitre = """
Dans toute entreprise employant au moins trois cents salariés et dans toute entreprise spécialisée dans
le recrutement, les employés chargés des missions de recrutement reçoivent une formation à la non-
discrimination à l'embauche au moins une fois tous les cinq ans.
"""

article_with_plenty_noise = """
Aucune personne ne peut être écartée d'une procédure de recrutement ou de l'accès à un stage ou à une
période de formation en entreprise, aucun salarié ne peut être sanctionné, licencié ni faire l'objet d'une
mesure discriminatoire, directe ou indirecte, notamment en matière de rémunération, au sens de l'article L.
3221-3, de mesures d'intéressement ou de distribution d'actions, de formation, de reclassement, d'affectation,
de qualification, de classification, de promotion professionnelle, d'horaires de travail, d'évaluation de la
performance, de mutation ou de renouvellement de contrat, ni de toute autre mesure mentionnée au II
de l'article 10-1 de la loi n° 2016-1691 du 9 décembre 2016 relative à la transparence, à la lutte contre la
corruption et à la modernisation de la vie économique, pour avoir signalé ou divulgué des informations dans
les conditions prévues aux articles 6 et 8 de la même loi.
Partie législative
Première partie : Les relations individuelles de travail
Livre Ier : Dispositions préliminaires
Titre III : Discriminations
Chapitre Ier : Champ d'application.
"""

cleaned_article_with_plenty_noise = """
Aucune personne ne peut être écartée d'une procédure de recrutement ou de l'accès à un stage ou à une
période de formation en entreprise, aucun salarié ne peut être sanctionné, licencié ni faire l'objet d'une
mesure discriminatoire, directe ou indirecte, notamment en matière de rémunération, au sens de l'article L.
3221-3, de mesures d'intéressement ou de distribution d'actions, de formation, de reclassement, d'affectation,
de qualification, de classification, de promotion professionnelle, d'horaires de travail, d'évaluation de la
performance, de mutation ou de renouvellement de contrat, ni de toute autre mesure mentionnée au II
de l'article 10-1 de la loi n° 2016-1691 du 9 décembre 2016 relative à la transparence, à la lutte contre la
corruption et à la modernisation de la vie économique, pour avoir signalé ou divulgué des informations dans
les conditions prévues aux articles 6 et 8 de la même loi.
"""

article_with_noise_both_side = """
Livre 2ème : Dispositions préliminaires
Titre V : Sécurité
Les dispositions du présent livre sont applicables aux employeurs de droit privé ainsi qu'à leurs salariés.
 
Elles sont également applicables au personnel des personnes publiques employé dans les conditions du
droit privé, sous réserve des dispositions particulières ayant le même objet résultant du statut qui régit ce
personnel.
Chapitre Ier : Champ d'application.
"""

cleaned_article_with_noise_both_side = """
Les dispositions du présent livre sont applicables aux employeurs de droit privé ainsi qu'à leurs salariés.
 
Elles sont également applicables au personnel des personnes publiques employé dans les conditions du
droit privé, sous réserve des dispositions particulières ayant le même objet résultant du statut qui régit ce
personnel.
"""