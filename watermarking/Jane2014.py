"""
Jane, O., Elbaşi, E., & İlk, H. G. (2014). 
Hybrid Non-Blind Watermarking Based on DWT and SVD. 
Journal of Applied Research and Technology, 12(4), 750–761. 
https://doi.org/10.1016/S1665-6423(14)70091-4
"""

from pywt import dwt2, idwt2

from Liu2002 import Watermarker as Liu2002Watermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2

    def __init__(self, scale_factor=0.05) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        self.svd_watermarker = Liu2002Watermarker(self.sf)

        LL, (HL, LH, HH) = dwt2(host, wavelet="haar")

        LL_ = self.svd_watermarker.add_watermark(LL, watermark)

        return idwt2((LL_, (HL, LH, HH)), wavelet="haar")

    def extract_watermark(self, watermarked_image):
        LL, (HL, LH, HH) = dwt2(watermarked_image, wavelet="haar")
        return self.svd_watermarker.extract_watermark(LL)
