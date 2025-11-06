import json

import pytest

from analyze import analyze_text, detect_dates, is_action_verb, keyword_analysis


@pytest.fixture
def sample_text():
    return (
        "John Doe\njohn@example.com\nExperience\n"  # header
        "- Led a team to improve efficiency by 30%.\n"
        "- Built an automation that saved 120 hours annually.\n"
        "Education\nBachelor of Science, 2020\n"
        "Skills\nPython, SQL, Excel\n"
    )


def test_is_action_verb():
    assert is_action_verb("Led")
    assert not is_action_verb("Was")


def test_detect_dates():
    text = "Worked from Jan 2020 - Dec 2021 and 03/2022"
    matches = detect_dates(text)
    assert any("Jan 2020" in m for m in matches)
    assert any("03/2022" in m for m in matches)


def test_keyword_analysis_with_jd():
    skills = ["python", "sql", "excel"]
    jd = "We need Python developers with SQL and Tableau experience"
    result = keyword_analysis(skills, jd)
    assert result["mode"] == "jd"
    assert "python" in result["matched"]
    assert "tableau" in result["missing"]


def test_analyze_text_scores(sample_text):
    result = analyze_text(sample_text)
    assert 0 <= result["overall_score"] <= 100
    for value in result["category_scores"].values():
        assert 0 <= value <= 25
    assert len(result["bullet_suggestions"]) <= 5
