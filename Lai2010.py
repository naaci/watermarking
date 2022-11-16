"""Lai, C.-C., & Tsai, C.-C. (2010). 
Digital Image Watermarking Using Discrete Wavelet Transform and Singular Value Decomposition. 
IEEE Transactions on Instrumentation and Measurement, 59(11), 3060–3063. 
https://doi.org/10.1109/TIM.2010.2066770
"""

from pywt import dwt2, idwt2

from Liu2002 import Watermarker as Liu2002Watermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2

    def __init__(self, scale_factor=.01) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        # 1)  Use one-level Haar DWT to decompose the cover imageAinto four subbands (i.e., LL, LH, HL, and HH).
        LL, (HL, LH, HH) = dwt2(host, wavelet='haar')

        # 2)  Apply SVD to LH and HL subbands, i.e.,Ak=UkSkVkT,k=1,2(1)wherekrepresents one of two subbands.
        self.svd_watermarker_1 = Liu2002Watermarker(self.sf)
        self.svd_watermarker_2 = Liu2002Watermarker(self.sf)

        # 3)  Divide  the  watermark  into  two  parts: W=W_1+W_2,where W_k denotes half of the watermark.
        watermark_1 = watermark.copy()
        watermark_1[watermark > watermark.mean()] = 0
        watermark_2 = watermark - watermark_1

        # 4)  Modify  the  singular  values  in  HL  and  LH  subbands  with half of the watermark image and then apply SVD to them,respectively,
        # 5)  Obtain the two sets of modified DWT coefficients, i.e.,A∗k=UkSkWVkT,k=1,2.(3)
        LH_ = self.svd_watermarker_1.add_watermark(LH, watermark_1)
        HL_ = self.svd_watermarker_2.add_watermark(HL, watermark_2)

        # 6)  Obtain  the  watermarked  imageAWby  performing  the  in-verse DWT using two sets of modified DWT coefficients andtwo sets of nonmodified DWT coefficients.

        return idwt2((LL, (HL_, LH_, HH)), wavelet='haar')

    def extract_watermark(self, watermarked_image):
        # 1)  Use  one-level  Haar  DWT  to  decompose  the  watermarked(possibly distorted) imageA∗Winto four subbands: LL, LH,HL, and HH.
        LL, (HL_, LH_, HH) = dwt2(watermarked_image, wavelet='haar')

        # 2)  Apply SVD to the LH and HL subbands, i.e.,A∗kW=U∗kS∗kWV∗kT,k=1,2(4)wherekrepresents one of two subbands.
        # 3)  Compute D∗k=UkWS∗kWVkTW,k=1,2.
        # 4)  Extract half of the watermark image from each subband, i.e.,W∗k=(D∗k−Sk)/α,k=1,2.(5)

        watermark_image_1 = self.svd_watermarker_1.extract_watermark(LH_)
        watermark_image_2 = self.svd_watermarker_2.extract_watermark(HL_)

        # 5)  Combine the results of Step 4 to obtain the embedded water-mark:W∗=W∗1+W∗2.
        return watermark_image_1 + watermark_image_2


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
