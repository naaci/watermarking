"""
Jain, C., Arora, S., &#38; Panigrahi, P. K. (2008). 
A Reliable SVD based Watermarking Scheme. May 2018, 1–8. 
preprint
http://arxiv.org/abs/0808.0309
"""

# from numpy.linalg import svd
from scipy.linalg import diagsvd, svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 1

    def __init__(self, scale_factor=.02) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        self.host = host
        w, h = host.shape
        self.U, s, self.Vh = svd(self.host, full_matrices=True)
        U_w, s_w, self.Vh_w = svd(watermark, full_matrices=True)
        P_w = U_w @ diagsvd(s_w, *watermark.shape)
        D = diagsvd(s, w, h) + self.sf * P_w
        return self.U @ D @ self.Vh

    def extract_watermark(self, watermarked):
        D = watermarked - self.host
        return (self.U.T.conj() @ D @ self.Vh.T.conj()) / self.sf @ self.Vh_w
