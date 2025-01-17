"""
Yavuz, E., & Telatar, Z. (2007). 
Improved SVD-DWT based digital image watermarking against watermark ambiguity. 
Proceedings of the 2007 ACM Symposium on Applied Computing - SAC ’07, 1051. 
https://doi.org/10.1145/1244002.1244232
"""

from numpy import log2, sqrt
from numpy import zeros_like
from pywt import wavedec2, waverec2

from Chandra2002 import Watermarker as chandraWatermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2**3
    SQUARE_IMAGE_ONLY = True

    def __init__(self, scale_factor=0.1) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        self.level = int(log2(host.shape[0] // watermark.shape[0]))

        self.svd_watermarker_1 = chandraWatermarker(self.sf)
        self.svd_watermarker_2 = chandraWatermarker(self.sf * 0.15)

        # 3rd level DWT is applied to the cover image and LL, HL, LH and HH subbands are obtained
        LL, (HL, self.LH, self.HH), *a = wavedec2(
            host, wavelet="haar", level=self.level
        )

        LL3_ = self.svd_watermarker_1.add_watermark(LL, watermark)
        HL3_ = self.svd_watermarker_2.add_watermark(HL, watermark)

        # Components of U matrix of the watermark are embedded into LH and HH subbands
        # self.U_w, s_w, self.Vh_w = svd(watermark, full_matrices=False)

        LH3_ = self.LH + self.svd_watermarker_1.U_w * self.sf
        HH3_ = self.HH + self.svd_watermarker_1.U_w * self.sf

        return waverec2(
            (LL3_, (HL3_, LH3_, HH3_), *a),
            wavelet="haar",
        )

    def extract_watermark(self, watermarked_image):
        LL, (HL, LH, HH), *a = wavedec2(
            watermarked_image, wavelet="haar", level=self.level
        )

        U_w_ = (LH + HH - self.LH - self.HH) / (2 * self.sf)
        ncc = (U_w_ * self.svd_watermarker_1.U_w).sum() / (
            sqrt((U_w_**2).sum()) * sqrt((self.svd_watermarker_1.U_w**2).sum())
        )

        if ncc > 0.2:
            return self.svd_watermarker_1.extract_watermark(LL)
            return self.svd_watermarker_2.extract_watermark(HL)
        else:
            return zeros_like(LL)

    def extract_watermarks(self, watermarked_image):
        LL, (HL, LH, HH), *a = wavedec2(
            watermarked_image, wavelet="haar", level=self.level
        )

        U_w_ = (LH + HH - self.LH - self.HH) / (2 * self.sf)
        ncc = (U_w_ * self.svd_watermarker_1.U_w).sum() / (
            sqrt((U_w_**2).sum()) * sqrt((self.svd_watermarker_1.U_w**2).sum())
        )

        if ncc > 0.2:
            return (
                self.svd_watermarker_1.extract_watermark(LL),
                self.svd_watermarker_2.extract_watermark(HL),
            )
        else:
            return zeros_like(LL), zeros_like(LL)
