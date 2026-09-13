"""Unit tests for Curriculum Planner parsing (no API calls)."""

import pytest

from agents.curriculum_planner import parse_roadmap_json


VALID_JSON = """
{
  "goal": "Learn Python closures",
  "total_weeks": 2,
  "weekly_hours": 5,
  "topics": [
    {
      "title": "Functions Review",
      "description": "Review first-class functions",
      "estimated_minutes": 45,
      "prerequisites": [],
      "status": "pending"
    },
    {
      "title": "Closures",
      "description": "Understand closures",
      "estimated_minutes": 60,
      "prerequisites": ["Functions Review"],
      "status": "pending"
    }
  ]
}
"""


def test_parse_valid_roadmap():
    roadmap = parse_roadmap_json(VALID_JSON)
    assert roadmap.goal == "Learn Python closures"
    assert roadmap.total_weeks == 2
    assert len(roadmap.topics) == 2
    assert roadmap.topics[1].prerequisites == ["Functions Review"]


def test_parse_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_roadmap_json("not json {")


def test_parse_missing_topics():
    with pytest.raises(ValueError, match="missing required field"):
        parse_roadmap_json('{"goal": "x", "total_weeks": 1}')


def test_parse_empty_topics():
    with pytest.raises(ValueError, match="non-empty list"):
        parse_roadmap_json(
            '{"goal": "x", "total_weeks": 1, "topics": []}'
        )


def test_parse_topic_missing_field():
    with pytest.raises(ValueError, match="Topic 0 missing"):
        parse_roadmap_json(
            '{"goal": "x", "total_weeks": 1, "topics": [{"title": "A"}]}'
        )
