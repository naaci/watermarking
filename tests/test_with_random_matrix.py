import sys
from importlib import import_module
from pathlib import Path

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

    # c = isclose(watermark, watermark_)
    # print("Watermarking Test result:")
    # print("method:".ljust(10), Watermarker.__module__)
    # print("max err:".ljust(10), abs(watermark - watermark_).max())
    # print("isclose:".ljust(10), f"{watermark[c].size / watermark.size:>.2%}")


def test_watermarks():
    for module in cwd.iterdir():
        if (
            module.name.startswith(".")
            or module.name.startswith("_")
            or module.is_dir()
            or not module.name.endswith(".py")
        ):
            continue
        module = module.name.removesuffix(".py")
        print(module)

        watermarker = import_module(module).Watermarker(0.01)
        _test_watermark(watermarker)


if __name__ == "__main__":
    test_watermarks()
