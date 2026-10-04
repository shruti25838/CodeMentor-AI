import pytest

from codeatlas.controllers.files_controller import _safe_join


def test_file_inside_root_resolves(tmp_path):
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "app.py"
    f.write_text("x")
    assert _safe_join(tmp_path, "src/app.py") == f.resolve()


def test_parent_traversal_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("x")
    assert _safe_join(root, "../outside.txt") is None


def test_absolute_path_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    assert _safe_join(root, str(outside)) is None


def test_empty_path_rejected(tmp_path):
    assert _safe_join(tmp_path, "") is None


def test_symlink_outside_root_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks need extra privileges on this machine")
    assert _safe_join(root, "link.txt") is None
