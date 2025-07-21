from pdf_to_docx.utils import make_output_dirs
from pdf_to_docx.utils.io_tools import move_docx
from pdf_to_docx.utils.time_tools import convert_hhmmss


def test_convert_hhmmss():
    assert convert_hhmmss(0) == (0, 0, 0)
    assert convert_hhmmss(59) == (0, 0, 59)
    assert convert_hhmmss(60) == (0, 1, 0)
    assert convert_hhmmss(3600) == (1, 0, 0)
    assert convert_hhmmss(3665) == (1, 1, 5)


def test_make_output_dirs(tmp_path):
    base = tmp_path / "out"
    subfolders = ["txt", "pdf"]

    paths = make_output_dirs(str(base), subfolders)

    for name in subfolders:
        p = base / name
        assert p.exists()
        assert paths[name] == str(p)


def test_move_docx(tmp_path):
    origin = tmp_path / "original.docx"
    dest = tmp_path / "moved.docx"
    origin.write_text("dummy content")

    move_docx(str(origin), str(dest))

    assert not origin.exists()
    assert dest.exists()
    assert dest.read_text() == "dummy content"
