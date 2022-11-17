from Ganic2004 import Watermarker

from numpy import asarray, isclose, random
watermarker = Watermarker(.01)
host = random.random((500, 600))
watermark = random.random(asarray(host.shape) // watermarker.IMAGE_TO_WATERMARK_RATIO)
watermark[watermark >= .5] = 1
watermark[watermark < .5] = 0
watermarked = watermarker.add_watermark(host, watermark)
watermark_ = watermarker.extract_watermark(watermarked)
c = isclose(watermark, watermark_)
print("max err:", abs(watermark - watermark_).max())
print(c.all() or f"{watermark[c].size / watermark.size:>.2%} True")
