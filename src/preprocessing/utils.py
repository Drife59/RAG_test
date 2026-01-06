def distance_levenshtein(s1, s2):
    """Calculate the distance between two strings.

    Distance means the number of operations needed to transform s1 into s2.
    - insertion: add a character to s1
    - deletion: remove a character from s1
    - substitution: replace a character in s1 with another
    """
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


def string_similarity(chaine1: str, chaine2: str) -> float:
    """Calculate the similarity in percentage between two strings."""
    distance = distance_levenshtein(chaine1, chaine2)
    max_length = max(len(chaine1), len(chaine2))
    similarite = 1.0 - (distance / max_length) if max_length != 0 else 1.0

    return similarite
