"""
Yavuz, E., & Telatar, Z. (2007). 
Improved SVD-DWT based digital image watermarking against watermark ambiguity. 
Proceedings of the 2007 ACM Symposium on Applied Computing - SAC ’07, 1051. 
https://doi.org/10.1145/1244002.1244232
"""

from numpy import isclose, log2, sqrt
from numpy.core.multiarray import empty_like
from numpy.core.numeric import zeros_like
from numpy.linalg import svd
from pywt import wavedec2, waverec2

from Chandra2002 import Watermarker as chandraWatermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2**3

    def __init__(self, scale_factor=.1) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        self.level = int(log2(host.shape[0]//watermark.shape[0]))

        self.svd_watermarker_1 = chandraWatermarker(self.sf)
        self.svd_watermarker_2 = chandraWatermarker(self.sf*.15)

        # 3rd level DWT is applied to the cover image and LL, HL, LH and HH subbands are obtained
        LL, (HL, self.LH, self.HH), *a = wavedec2(host,
                                                  wavelet='haar',
                                                  level=self.level)

        LL3_ = self.svd_watermarker_1.add_watermark(LL, watermark)
        HL3_ = self.svd_watermarker_2.add_watermark(HL, watermark)

        # Components of U matrix of the watermark are embedded into LH and HH subbands
        # self.U_w, s_w, self.Vh_w = svd(watermark, full_matrices=False)

        LH3_ = self.LH + self.svd_watermarker_1.U_w * self.sf
        HH3_ = self.HH + self.svd_watermarker_1.U_w * self.sf

        return waverec2(
            (LL3_, (HL3_, LH3_, HH3_), *a),
            wavelet='haar',
        )

    def extract_watermark(self, watermarked_image):
        LL, (HL, LH, HH), *a = wavedec2(watermarked_image,
                                        wavelet='haar',
                                        level=self.level)

        U_w_ = (LH + HH - self.LH - self.HH) / (2*self.sf)
        ncc = (U_w_ * self.svd_watermarker_1.U_w).sum() / \
            (sqrt((U_w_**2).sum()) * sqrt((self.svd_watermarker_1.U_w**2).sum()))

        if ncc > .2:
            return self.svd_watermarker_1.extract_watermark(LL)
            return self.svd_watermarker_2.extract_watermark(HL)
        else:
            return zeros_like(LL)

    def extract_watermarks(self, watermarked_image):
        LL, (HL, LH, HH), *a = wavedec2(watermarked_image,
                                        wavelet='haar',
                                        level=self.level)

        U_w_ = (LH + HH - self.LH - self.HH) / (2*self.sf)
        ncc = (U_w_ * self.svd_watermarker_1.U_w).sum() / \
            (sqrt((U_w_**2).sum()) * sqrt((self.svd_watermarker_1.U_w**2).sum()))

        if ncc > .2:
            return (self.svd_watermarker_1.extract_watermark(LL),
                    self.svd_watermarker_2.extract_watermark(HL))
        else:
            return zeros_like(LL), zeros_like(LL)


if __name__ == "__main__":
    from numpy import asarray, ceil, floor, isclose, random
    watermarker = Watermarker(.01)
    host = random.random((512, 512))
    watermark = random.random(
        ceil(asarray(host.shape) /
             watermarker.IMAGE_TO_WATERMARK_RATIO).astype(int))
    watermark[watermark >= .5] = 1
    watermark[watermark < .5] = 0
    watermarked = watermarker.add_watermark(host, watermark)
    watermark_ = watermarker.extract_watermark(watermarked)
    c = isclose(watermark, watermark_)
    print("max err:", abs(watermark - watermark_).max())
    print(c.all() or f"{watermark[c].size / watermark.size:>.2%} True")
