from numpy import asarray, isclose, random

from Ganic2004 import Watermarker

watermarker = Watermarker(.01)
host = random.random((500, 600))
watermark = random.random(asarray(host.shape) // watermarker.IMAGE_TO_WATERMARK_RATIO)
watermark[watermark >= .5] = 1
watermark[watermark < .5] = 0
watermarked = watermarker.add_watermark(host, watermark)
watermark_ = watermarker.extract_watermark(watermarked)
c = isclose(watermark, watermark_)
print("Watermarking Test result:")
print("method:".ljust(10),Watermarker.__module__)
print("max err:".ljust(10),abs(watermark - watermark_).max())
print("isclose:".ljust(10),f"{watermark[c].size / watermark.size:>.2%}")
