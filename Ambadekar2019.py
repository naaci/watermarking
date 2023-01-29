"""
Ambadekar, S. P., Jain, J., & Khanapuri, J. (2019). 
Digital Image Watermarking Through Encryption and DWT for Copyright Protection. 
In Advances in Intelligent Systems and Computing (Vol. 727, pp. 187–195). Springer Singapore. 
https://doi.org/10.1007/978-981-10-8863-6_19"""

from pywt import dwt2, idwt2


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 1

    def __init__(self, scale_factor=2**-6) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        self.LL, (self.HL, self.LH, self.HH) = dwt2(host, wavelet='haar')
        LL_w, (HL_w, LH_w, HH_w) = dwt2(watermark, wavelet='haar')

        LL_ = self.LL*(1-self.sf) + LL_w*self.sf
        HL_ = self.HL*(1-self.sf) + HL_w*self.sf
        LH_ = self.LH*(1-self.sf) + LH_w*self.sf
        HH_ = self.HH*(1-self.sf) + HH_w*self.sf

        return idwt2((LL_, (HL_, LH_, HH_)), wavelet='haar')

    def extract_watermark(self, watermarked_image):
        LL_, (HL_, LH_, HH_) = dwt2(watermarked_image, wavelet='haar')
        return idwt2((
            (LL_-self.LL*(1-self.sf))/self.sf,
            ((HL_-self.HL*(1-self.sf))/self.sf,
             (LH_-self.LH*(1-self.sf))/self.sf,
             (HH_-self.HH*(1-self.sf))/self.sf,),
        ), wavelet='haar')
