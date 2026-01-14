import json
from dataclasses import asdict, dataclass


@dataclass
class ArticleEvaluation:
    id_article: str
    pertinent: bool
    justification: str

    @classmethod
    def from_dict(cls, data: dict) -> "ArticleEvaluation":
        return cls(
            id_article=data["id_article"],
            pertinent=data["pertinent"],
            justification=data["justification"]
        )
    

def filter_context(evaluated_contexts: list[ArticleEvaluation]) -> list[ArticleEvaluation]:
    pertinent_count = 0

    non_pertinent_context_docs = []
    for evaluated_context in evaluated_contexts:
        if evaluated_context.pertinent:
            pertinent_count += 1
        else:
            non_pertinent_context_docs.append(evaluated_context)

    with open("non_pertinent_context.txt", "w", encoding="utf-8") as f:
        context_as_dicts = [asdict(context) for context in non_pertinent_context_docs]
        f.write(json.dumps(context_as_dicts, indent=2, ensure_ascii=False))

    print(f"pertinent context count: {pertinent_count}")
    print(f"filtered context count: {len(evaluated_contexts) - pertinent_count}")
    if len(evaluated_contexts) > 0:
        print(f"percentage pertinent context: {pertinent_count / len(evaluated_contexts)}")

    return [context for context in evaluated_contexts if context.pertinent]