"""Sample-database downloads: a bad or stalled download must never reach the app."""

import http.server
import shutil
import socket
import threading

import pytest

from text2sql import samples


def test_a_download_that_is_not_a_database_never_replaces_the_target(tmp_path, monkeypatch):
    def error_page(url, dest, log):
        dest.write_bytes(b"<!doctype html><title>Service unavailable</title>")
        return dest

    monkeypatch.setattr(samples, "_fetch", error_page)
    out = tmp_path / "data" / "chinook.db"
    with pytest.raises(ValueError, match="not a usable SQLite database"):
        samples.download("chinook", out, log=lambda _msg: None)
    assert list(out.parent.iterdir()) == []  # neither the file nor a leftover .part


def test_a_good_download_is_checked_and_moved_into_place(tmp_path, shop_db, monkeypatch):
    def copy_shop(url, dest, log):
        shutil.copy(shop_db, dest)
        return dest

    monkeypatch.setattr(samples, "_fetch", copy_shop)
    out = tmp_path / "data" / "chinook.db"
    assert samples.download("chinook", out, log=lambda _msg: None) == out
    assert "customers" in samples.check_database(out)
    assert [p.name for p in out.parent.iterdir()] == ["chinook.db"]


def test_an_empty_database_is_rejected(tmp_path):
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="no tables"):
        samples.check_database(empty)


class _ShortBody(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - the name http.server looks for
        self.send_response(200)
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b"x" * 10)  # then the connection closes

    def log_message(self, *args):
        pass


def test_a_download_cut_off_early_is_an_error(tmp_path):
    server = http.server.HTTPServer(("127.0.0.1", 0), _ShortBody)
    threading.Thread(target=server.handle_request, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/chinook.sqlite"
    try:
        with pytest.raises(OSError, match="stopped early"):
            samples._fetch(url, tmp_path / "part", log=lambda _msg: None)
    finally:
        server.server_close()


def test_a_stalled_download_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(samples, "TIMEOUT_S", 0.2)
    with socket.socket() as silent:  # accepts the connection, never answers
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        url = f"http://127.0.0.1:{silent.getsockname()[1]}/chinook.sqlite"
        with pytest.raises(OSError):
            samples._fetch(url, tmp_path / "part", log=lambda _msg: None)
