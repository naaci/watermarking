"""
Zhu, T., Qu, W., &#38; Cao, W. (2022). 
An optimized image watermarking algorithm based on SVD and IWT. 
The Journal of Supercomputing, 78(1), 222–237. 
https://doi.org/10.1007/s11227-021-03886-2
"""

from pywt import dwt2, idwt2

from Sun2002 import Watermarker as SunWatermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2

    def __init__(self, scale_factor=2**(-4)) -> None:
        self.sf = scale_factor
        self.svd_watermarker = SunWatermarker(self.sf)
        self.IMAGE_TO_WATERMARK_RATIO *= self.svd_watermarker.IMAGE_TO_WATERMARK_RATIO

    def add_watermark(self, host, watermark):
        LL, H = dwt2(host, wavelet="haar")
        LL_ = self.svd_watermarker.add_watermark(LL, watermark)
        return idwt2((LL_, H), wavelet="haar")

    def extract_watermark(self, watermarked_image):
        LL_, _ = dwt2(watermarked_image, wavelet="haar")
        return self.svd_watermarker.extract_watermark(LL_)


if __name__ == "__main__":
    from numpy import asarray, isclose, random
    watermarker = Watermarker(.01)
    host = random.random((500, 600))
    watermark = random.random(
        asarray(host.shape) // watermarker.IMAGE_TO_WATERMARK_RATIO)
    watermark[watermark >= .5] = 1
    watermark[watermark < .5] = 0
    watermarked = watermarker.add_watermark(host, watermark)
    watermark_ = watermarker.extract_watermark(watermarked)
    c = isclose(watermark, watermark_)
    print("max err:", abs(watermark - watermark_).max())
    print(c.all() or f"{watermark[c].size / watermark.size:>.2%} True")
