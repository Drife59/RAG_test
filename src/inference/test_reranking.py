from src.inference.reranking import filter_context


def test_empty_list():
    result = filter_context([])
    assert result == []


def test_all_pertinent():
    evaluations = [
        {"id_article": "ID1", "pertinent": True, "justification": "..."},
        {"id_article": "ID2", "pertinent": True, "justification": "..."},
    ]
    result = filter_context(evaluations)
    assert len(result) == 2
    assert all(context["pertinent"] for context in result)


def test_none_pertinent():
    evaluations = [
        {"id_article": "ID1", "pertinent": False, "justification": "..."},
        {"id_article": "ID2", "pertinent": False, "justification": "..."},
    ]
    result = filter_context(evaluations)
    assert len(result) == 0


def test_mixed_pertinent():
    evaluations = [
        {"id_article": "ID1", "pertinent": True, "justification": "..."},
        {"id_article": "ID2", "pertinent": False, "justification": "..."},
        {"id_article": "ID3", "pertinent": True, "justification": "..."},
    ]
    result = filter_context(evaluations)
    assert len(result) == 2
    assert all(context["pertinent"] for context in result)


def test_single_element_pertinent():
    """Test avec un seul élément pertinent."""
    evaluations = [
        {"id_article": "ID1", "pertinent": True, "justification": "..."},
    ]
    result = filter_context(evaluations)
    assert len(result) == 1
    assert result[0]["pertinent"]


def test_single_element_not_pertinent():
    """Test avec un seul élément non pertinent."""
    evaluations = [
        {"id_article": "ID1", "pertinent": False, "justification": "..."},
    ]
    result = filter_context(evaluations)
    assert len(result) == 0
