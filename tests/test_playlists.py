from src.youtube_utils import (
    parse_playlist_data,
    parse_playlist_search_entries,
    wants_playlists,
)


def test_playlist_intent_is_detected():
    assert wants_playlists("python playlist")
    assert wants_playlists("Best DSA PlayLists for placements")
    assert wants_playlists("react play list")
    assert not wants_playlists("explain transformers simply")
    assert not wants_playlists("")


def test_playlist_search_keeps_only_playlists():
    entries = [
        {"id": "UCabcdefghijklmnopqrstuv", "title": "A channel", "url": "https://www.youtube.com/channel/UCabc"},
        {"id": "dQw4w9WgXcQ", "title": "A video", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        {
            "id": "PLabc123",
            "title": "Python Full Course",
            "uploader": "Teacher",
            "url": "https://www.youtube.com/playlist?list=PLabc123",
            "thumbnails": [{"url": "small"}, {"url": "big"}],
        },
        {"id": "PLabc123", "title": "duplicate"},
        None,
    ]
    out = parse_playlist_search_entries(entries, 5)
    assert [p["playlist_id"] for p in out] == ["PLabc123"]
    assert out[0]["title"] == "Python Full Course"
    assert out[0]["channel"] == "Teacher"
    assert out[0]["thumbnail"] == "big"
    assert out[0]["url"] == "https://www.youtube.com/playlist?list=PLabc123"


def test_playlist_search_respects_limit_and_list_param():
    entries = [
        {"id": "x", "title": f"mix {i}", "url": f"https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL{i:03d}abc"}
        for i in range(5)
    ]
    assert len(parse_playlist_search_entries(entries, 3)) == 3


def test_playlist_data_skips_private_deleted_and_invalid_entries():
    data = {
        "id": "PL1",
        "title": "Course",
        "uploader": "Chan",
        "playlist_count": 4,
        "entries": [
            {"id": "dQw4w9WgXcQ", "title": "Lesson 1", "duration": 65.0, "channel": "C"},
            {"id": "dQw4w9WgXcQ", "title": "duplicate"},
            {"id": "bad", "title": "not a video id"},
            {"id": "aaaaaaaaaaa", "title": "[Private video]"},
            {"id": "bbbbbbbbbbb", "title": "[Deleted video]"},
            None,
            {"id": "ccccccccccc", "title": "Lesson 2", "duration": None},
        ],
    }
    out = parse_playlist_data(data)
    assert out["title"] == "Course" and out["channel"] == "Chan" and out["video_count"] == 4
    assert [e["video_id"] for e in out["entries"]] == ["dQw4w9WgXcQ", "ccccccccccc"]
    assert out["entries"][0]["duration"] == 65 and out["entries"][1]["duration"] == 0


def test_playlist_data_limit():
    data = {"entries": [{"id": f"{i:011d}", "title": f"v{i}"} for i in range(10)]}
    assert len(parse_playlist_data(data, 4)["entries"]) == 4
