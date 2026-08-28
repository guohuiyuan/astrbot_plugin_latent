from __future__ import annotations

import pytest

from latent.utils import (
    extract_tag_text,
    parse_options,
    resolve_enum,
    strip_command,
    to_int,
    try_nonnegative_int,
)


def test_strip_command_removes_leading_command():
    assert strip_command("/生图 1girl, solo", {"生图", "draw"}) == "1girl, solo"
    assert strip_command("/draw 1girl", {"draw"}) == "1girl"
    assert strip_command("1girl", {"draw"}) == "1girl"
    assert strip_command("/生图", {"生图"}) == ""
    assert strip_command("", {"生图"}) == ""


def test_parse_options_extracts_and_removes_options():
    text, options = parse_options(
        "1girl, long hair --steps 10 --resolution portrait --negative lowres, bad anatomy"
    )
    assert text == "1girl, long hair"
    assert options["steps"] == "10"
    assert options["resolution"] == "portrait"
    assert options["negative"] == "lowres, bad anatomy"


def test_parse_options_handles_quoted_negative():
    text, options = parse_options('a cat --negative "lowres, jpeg"')
    assert text == "a cat"
    assert options["negative"] == "lowres, jpeg"


def test_parse_options_last_option_without_trailing():
    text, options = parse_options("1girl --count 3")
    assert text == "1girl"
    assert options["count"] == "3"


def test_resolve_enum_returns_default_for_unknown():
    assert resolve_enum("square", {"square", "portrait"}, "portrait") == "square"
    assert resolve_enum("bogus", {"square", "portrait"}, "portrait") == "portrait"
    assert resolve_enum("", {"square", "portrait"}, "portrait") == "portrait"


def test_to_int_clamps():
    assert to_int("12", 12, 8, 16) == 12
    assert to_int("20", 12, 8, 16) == 16
    assert to_int("3", 12, 8, 16) == 8
    assert to_int("bogus", 12, 8, 16) == 12


def test_try_nonnegative_int():
    assert try_nonnegative_int("42") == 42
    assert try_nonnegative_int("0") == 0
    assert try_nonnegative_int("-1") is None
    assert try_nonnegative_int("") is None
    assert try_nonnegative_int(None) is None


def test_extract_tag_text_json_array():
    assert extract_tag_text('["masterpiece", "best quality", "1girl"]') == [
        "masterpiece",
        "best_quality",
        "1girl",
    ]


def test_extract_tag_text_plain_line():
    assert extract_tag_text("masterpiece, best quality, 1girl, long hair") == [
        "masterpiece",
        "best_quality",
        "1girl",
        "long_hair",
    ]


def test_extract_tag_text_strips_fence_and_noise():
    text = "```danbooru\nHere are the tags:\nmasterpiece, best quality, sunset\n```"
    assert extract_tag_text(text) == ["masterpiece", "best_quality", "sunset"]
