"""
Ganic, E. (2005). 
Robust embedding of visual watermarks using discrete wavelet transform and singular value decomposition. 
Journal of Electronic Imaging, 14(4), 043004. 
https://doi.org/10.1117/1.2137650
"""

from pywt import dwt2, idwt2

from Jain2008 import Watermarker as JainWatermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2

    def __init__(self, scale_factor=.05) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        LL, (HL, LH, HH) = dwt2(host, wavelet='haar')

        self.svd_watermarker_1 = JainWatermarker(self.sf)
        self.svd_watermarker_2 = JainWatermarker(self.sf / 10)
        self.svd_watermarker_3 = JainWatermarker(self.sf / 10)
        self.svd_watermarker_4 = JainWatermarker(self.sf / 10)

        LL_ = self.svd_watermarker_1.add_watermark(LL, watermark)
        HL_ = self.svd_watermarker_2.add_watermark(HL, watermark)
        LH_ = self.svd_watermarker_3.add_watermark(LH, watermark)
        HH_ = self.svd_watermarker_4.add_watermark(HH, watermark)

        return idwt2((LL_, (HL_, LH_, HH_)), wavelet='haar')

    def extract_watermark(self, watermarked_image):
        LL, (HL, LH, HH) = dwt2(watermarked_image, wavelet='haar')
        return self.svd_watermarker_1.extract_watermark(LL)
        return self.svd_watermarker_2.extract_watermark(HL)
        return self.svd_watermarker_3.extract_watermark(LH)
        return self.svd_watermarker_4.extract_watermark(HH)

    def extract_watermarks(self, watermarked_image):
        LL, (HL, LH, HH) = dwt2(watermarked_image, wavelet='haar')
        return (
            self.svd_watermarker_1.extract_watermark(LL),
            self.svd_watermarker_2.extract_watermark(HL),
            self.svd_watermarker_3.extract_watermark(LH),
            self.svd_watermarker_4.extract_watermark(HH),
        )


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
