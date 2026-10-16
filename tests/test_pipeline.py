from unittest.mock import patch


def test_playlist_processes_items_independently():
    from src.pipeline import process_playlist

    fake_results = {
        "a": {"meta": {"video_id": "a", "title": "A"}},
        "b": RuntimeError("bad video"),
        "c": {"meta": {"video_id": "c", "title": "C"}},
    }

    def fake_process(video_id, progress_callback=None):
        value = fake_results[video_id]
        if isinstance(value, Exception):
            raise value
        return value

    with patch("src.pipeline.extract_target", return_value=("playlist", "PL123")), patch(
        "src.pipeline.get_playlist_video_ids", return_value=["a", "b", "c"]
    ), patch("src.pipeline.process_video", side_effect=fake_process):
        result = process_playlist("https://youtube.com/playlist?list=PL123")

    assert result["processed"] == 2
    assert result["requested"] == 3
    assert len(result["failures"]) == 1
