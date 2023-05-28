"""
Mohammad, A. A., Alhaj, A., & Shaltaf, S. (2008). 
An improved SVD-based watermarking scheme for protecting rightful ownership. 
Signal Processing, 88(9), 2158–2180. 
https://doi.org/10.1016/j.sigpro.2008.02.015
"""

# from numpy.linalg import svd
from scipy.linalg import diagsvd, svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 1

    def __init__(self, scale_factor=0.02) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        self.U, s, self.Vh = svd(host, full_matrices=True)
        self.S = diagsvd(s, *host.shape[:2])
        D = self.S + watermark * self.sf
        return self.U @ D @ self.Vh

    def extract_watermark(self, watermarked_image):
        D = self.U.T.conj() @ watermarked_image @ self.Vh.T.conj()
        return (D - self.S) / self.sf
