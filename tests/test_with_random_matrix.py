import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

from numpy import asarray, isclose, random

cwd = Path(__file__).parent.parent / "watermarking"
sys.path.append(str(cwd))


def _test_watermark(watermarker):
    if hasattr(watermarker, "SQUARE_IMAGE_ONLY"):
        host = random.random((64, 64))
    else:
        host = random.random((64, 512))

    watermark = random.random(
        asarray(host.shape) // watermarker.IMAGE_TO_WATERMARK_RATIO
    )

    if hasattr(watermarker, "BINARY_WATERMARK"):
        watermark[watermark >= 0.5] = 1
        watermark[watermark < 0.5] = 0

    watermarked = watermarker.add_watermark(host, watermark)
    watermark_ = watermarker.extract_watermark(watermarked)
    assert isclose(watermark, watermark_).all(), watermarker.__module__

    # print("Watermarking Test result:")
    # print("method:".ljust(10), Watermarker.__module__)
    # print("max err:".ljust(10), abs(watermark - watermark_).max())
    # print("isclose:".ljust(10), f"{watermark[c].size / watermark.size:>.2%}")


def test_watermarks():
    for method in cwd.glob("*.py"):
        if method.name == "__init__.py":
            continue

        print(method.name)

        _test_watermark(
            SourceFileLoader(method.name.removesuffix(".py"), method.as_posix())
            .load_module()
            .Watermarker(0.01)
        )


if __name__ == "__main__":
    test_watermarks()
