from __future__ import annotations

__all__ = ["generate_from_file"]


def generate_from_file(*args, **kwargs):
    from src.generator import generate_from_file as _generate_from_file

    return _generate_from_file(*args, **kwargs)
