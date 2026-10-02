from movie_recommender.recommender import load_catalog, recommend


def test_recommendations_exclude_the_input_and_are_ranked():
    catalog = load_catalog()
    results = recommend("Spirited Away", catalog, limit=3)
    assert len(results) == 3
    assert all(item["title"] != "Spirited Away" for item in results)


def test_unknown_title_has_a_helpful_error():
    catalog = load_catalog()
    try:
        recommend("Missing", catalog)
    except ValueError as error:
        assert "unknown title" in str(error)
    else:
        raise AssertionError("expected an unknown-title error")
