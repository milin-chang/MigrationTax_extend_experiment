"""Use installed Times New Roman faces and record the actual selected files."""

from hashlib import sha256
from pathlib import Path


def times_new_roman():
    from matplotlib import font_manager

    directory = Path("/mnt/c/Windows/Fonts")
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = directory / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    faces = []
    for weight, style in (
        ("normal", "normal"),
        ("bold", "normal"),
        ("normal", "italic"),
        ("bold", "italic"),
    ):
        requested = font_manager.FontProperties(
            family="Times New Roman", weight=weight, style=style
        )
        selected = Path(font_manager.findfont(requested, fallback_to_default=False))
        family = font_manager.FontProperties(fname=str(selected)).get_name()
        if family != "Times New Roman":
            raise ValueError("The selected font is not Times New Roman")
        faces.append(
            dict(
                family=family,
                weight=weight,
                style=style,
                file=str(selected),
                sha256=sha256(selected.read_bytes()).hexdigest(),
            )
        )
    return faces
