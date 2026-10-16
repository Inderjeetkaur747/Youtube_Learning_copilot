import pytest

from src.youtube_utils import (
    extract_playlist_id,
    extract_target,
    extract_video_id,
    format_timestamp,
    is_playlist_url,
    parse_json3,
)


@pytest.mark.parametrize(
    "value",
    [
        "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "https://youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_extract_video_id(value):
    assert extract_video_id(value) == "dQw4w9WgXcQ"


def test_invalid_id():
    with pytest.raises(ValueError):
        extract_video_id("definitely-invalid")


def test_timestamp():
    assert format_timestamp(155) == "02:35"
    assert format_timestamp(3725) == "1:02:05"


def test_json3():
    data = {
        "events": [
            {
                "tStartMs": 1000,
                "dDurationMs": 2000,
                "segs": [{"utf8": "Hello "}, {"utf8": "world"}],
            }
        ]
    }
    assert parse_json3(data) == [{"text": "Hello world", "start": 1.0, "end": 3.0}]


def test_playlist_detection():
    url = "https://www.youtube.com/playlist?list=PL1234567890"
    assert is_playlist_url(url) is True
    assert extract_playlist_id(url) == "PL1234567890"
    assert extract_target(url) == ("playlist", "PL1234567890")


def test_video_with_playlist_parameter_is_still_a_video():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890"
    assert extract_target(url) == ("video", "dQw4w9WgXcQ")


def test_pick_track_prefers_spoken_language_and_human_captions():
    from src.youtube_utils import _pick_track

    info = {
        "language": "fr",
        "subtitles": {
            "en": [{"ext": "json3", "url": "en-human"}],
            "fr": [{"ext": "json3", "url": "fr-human"}],
        },
        "automatic_captions": {"fr": [{"ext": "json3", "url": "fr-auto"}]},
    }
    assert _pick_track(info) == ("fr", "fr-human")


def test_pick_track_prefers_human_over_auto_when_language_unknown():
    from src.youtube_utils import _pick_track

    info = {
        "subtitles": {"de": [{"ext": "json3", "url": "de-human"}]},
        "automatic_captions": {"en": [{"ext": "json3", "url": "en-auto"}]},
    }
    assert _pick_track(info) == ("de", "de-human")
