"""
Lagzian, S., Soryani, M., & Fathy, M. (2011). 
A New Robust Watermarking Scheme Based on RDWT-SVD. 
International Journal of Intelligent Information Processing, 2(1), 22–29. 
https://doi.org/10.4156/ijiip.vol2.issue1.3
"""

# from numpy.linalg import svd
from scipy.linalg import diagsvd, svd
from pywt import dwt2, idwt2


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 2

    def __init__(self, scale_factor=.02) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        LL, H = dwt2(host, wavelet='haar')
        self.U, s, self.Vh = svd(LL, full_matrices=False)
        self.U_w, s_w, self.Vh_w = svd(watermark, full_matrices=False)
        LL_ = self.U * s_w @ self.Vh
        return idwt2((LL_, H), wavelet='haar')

    def extract_watermark(self, watermarked_image):
        LL, H = dwt2(watermarked_image, wavelet='haar')
        s_w_ = svd(LL, compute_uv=False)
        return self.U_w * s_w_ @ self.Vh_w


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
